"""SM90 SS/RS numerical and CUDA Graph smoke test; run with torchrun.

Requires a built DeepGEMM extension, CUDA >= 12.9, and local NVLink peers.
Defaults are intentionally small; override shapes for production-size coverage.
This test is not run by the CPU-only port regression suite.
"""
import argparse
import os
import torch
import torch.distributed as dist
import deep_gemm as dg
from deep_gemm.mega import mega_moe_pre_dispatch_sm90


def decode(q, sf):
    q = q.to(torch.uint8)
    codes = torch.stack((q & 15, q >> 4), -1).flatten(-2).long()
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=q.device)
    values = lut[codes & 7] * torch.where((codes & 8) != 0, -1., 1.)
    return values * sf.repeat_interleave(32, -1)


def activation(gate, up, kind, clamp):
    if kind == 'situ':
        return 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * 25.0 * torch.tanh(up / 25.0)
    return torch.nn.functional.silu(gate.clamp(max=clamp)) * up.clamp(-clamp, clamp)


def reference(buf, w1, w2, n, activation_kind, clamp, world, rank):
    x = (buf.x[:n].float().reshape(n, -1, 128) * buf.x_sf[:n, :, None]).reshape(n, -1)
    def gather(t):
        result = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(result, t.contiguous())
        return torch.cat(result)
    all_x, routes, weights = gather(x), gather(buf.topk_idx[:n]), gather(buf.topk_weights[:n])
    result = torch.zeros_like(all_x)
    experts_per_rank = w1.size(0)
    for expert in range(experts_per_rank):
        tokens, slots = torch.where(routes == rank * experts_per_rank + expert)
        if tokens.numel() == 0:
            continue
        l1 = all_x[tokens] @ w1[expert].T
        gate, up = l1.chunk(2, -1)
        activated = activation(gate, up, activation_kind, clamp)
        weighted = activated * weights[tokens, slots, None]
        groups = weighted.reshape(weighted.size(0), -1, 64)
        scale = groups.abs().amax(-1).clamp(min=1e-10) * (1.0 / 448.0)
        quantized = (groups / scale[..., None]).to(torch.float8_e4m3fn).float()
        dequantized = (quantized * scale[..., None]).reshape_as(weighted)
        contribution = (dequantized @ w2[expert].T).to(torch.bfloat16).float()
        result.index_add_(0, tokens, contribution)
    dist.all_reduce(result)
    return result[rank*n:(rank+1)*n].to(torch.bfloat16)


def run(args):
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError('This test requires SM90')
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        if args.experts <= 0 or not (0 < args.topk <= 32) or args.experts % world or args.topk > args.experts:
            raise ValueError('positive experts must divide world size and topk must be <= min(experts, 32)')
        if args.hidden <= 0 or args.hidden % 512 or args.intermediate % 256 or not (0 < args.intermediate <= 4096):
            raise ValueError('RS requires hidden % 512 == 0 and intermediate % 256 == 0, <= 4096')
        batches = [int(x) for x in args.batches.split(',')]
        if not batches or min(batches) <= 0 or args.repeats < 1:
            raise ValueError('Provide positive batches and repeats')
        torch.manual_seed(args.seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        e, h, ih = args.experts // world, args.hidden, args.intermediate
        q1 = torch.randint(0, 256, (e, 2*ih, h//2), dtype=torch.uint8, device='cuda')
        q2 = torch.randint(0, 256, (e, h, ih//2), dtype=torch.uint8, device='cuda')
        sf1 = torch.full((e, 2*ih, h//32), 1/16, device='cuda')
        sf2 = torch.full((e, h, ih//32), 1/16, device='cuda')
        raw = ((q1, sf1), (q2, sf2))
        w1, w2 = decode(*raw[0]), decode(*raw[1])
        packed = {
            'ss': dg.transform_weights_for_mega_moe_sm90_fp4_ss(*raw),
            'rs': dg.transform_weights_for_mega_moe_sm90_fp4_rs(*raw),
        }
        operators = {'ss': dg.fp8_fp4_mega_moe_ss, 'rs': dg.fp8_fp4_mega_moe_rs}
        buf = dg.get_symm_buffer_for_mega_moe(
            dist.group.WORLD, args.experts, max(batches), args.topk, h, ih,
            mma_type='fp8xfp4', activation=args.activation)
        checks = 0
        for n in batches:
            x = torch.randn(n, h, device='cuda', dtype=torch.bfloat16)
            # Distinct experts per token. Invalid routes cover zero-output slots.
            topk = torch.rand(n, args.experts, device='cuda').topk(
                args.topk, dim=-1).indices.to(torch.int32)
            if args.routing == 'skewed':
                topk = torch.arange(
                    args.topk, device='cuda', dtype=torch.int32
                ).expand(n, -1).clone()
            if args.routing == 'invalid':
                topk.fill_(-1)
            else:
                topk[::7, 0] = -1
            weights = torch.rand(n, args.topk, device='cuda', dtype=torch.float32)
            weights /= weights.sum(-1, keepdim=True)
            mega_moe_pre_dispatch_sm90(x, topk, weights, buf.x, buf.x_sf,
                                      buf.topk_idx, buf.topk_weights, n)
            expected = reference(buf, w1, w2, n, args.activation, args.clamp, world, rank)
            kwargs = dict(activation=args.activation,
                          activation_clamp=None if args.activation == 'situ' else args.clamp,
                          activation_alpha=4., activation_beta=25., num_sms=args.num_sms)
            outputs, baselines, graphs = {}, {}, {}
            for backend in ('ss', 'rs'):
                y = torch.empty((n,h), device='cuda', dtype=torch.bfloat16)
                op = operators[backend]
                for _ in range(2):
                    op(y, *packed[backend], buf, **kwargs)
                torch.cuda.synchronize()
                baseline = y.clone()
                error = (baseline.float()-expected.float()).square().mean().sqrt()
                norm = expected.float().square().mean().sqrt().clamp(min=1e-6)
                relative_rmse = (error/norm).item()
                if not torch.isfinite(baseline).all() or relative_rmse > args.max_relative_rmse:
                    raise AssertionError(f'{backend} B{n}: relative RMSE {relative_rmse}')
                if args.routing == 'invalid' and torch.count_nonzero(baseline).item():
                    raise AssertionError(f'{backend} invalid routes must produce zero')
                for _ in range(args.repeats):
                    op(y, *packed[backend], buf, **kwargs)
                    torch.testing.assert_close(y, baseline, rtol=0, atol=0)
                    checks += 1
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with torch.cuda.graph(graph):
                    op(y, *packed[backend], buf, **kwargs)
                graphs[backend], outputs[backend], baselines[backend] = graph, y, baseline
                if rank == 0:
                    print(f'{backend} B{n}: relative_rmse={relative_rmse:.6g}', flush=True)
            # Interleave both metadata interpretations on exactly the same workspace.
            for _ in range(args.repeats):
                for backend in ('ss','rs'):
                    graphs[backend].replay()
                    torch.testing.assert_close(outputs[backend], baselines[backend], rtol=0, atol=0)
                    checks += 1
            dist.barrier()
        buf.destroy()
        if rank == 0:
            print(f'PASS: {checks} eager/graph repeat checks per rank; world_size={world}', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experts', type=int, default=8)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--intermediate', type=int, default=256)
    parser.add_argument('--batches', default='1,48,256,640')
    parser.add_argument('--activation', choices=['swiglu','situ'], default='swiglu')
    parser.add_argument('--clamp', type=float, default=10.)
    parser.add_argument('--routing', choices=['uniform','skewed','invalid'], default='uniform')
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--num-sms', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--max-relative-rmse', type=float, default=.05)
    run(parser.parse_args())
