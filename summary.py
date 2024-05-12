# TODO:Show network structure, GFLOPS, Params, memory, speed, FPS
import time
import torch
from prettytable import PrettyTable
from thop import clever_format, profile
from torch.backends import cudnn
from model.build_model.Build_MultiSenseSeg import Build_MultiSenseSeg


def compute_speed(model, input_size, device, n, operations='* & +', iteration=100):
    global v_mem
    input_size = [1] + list(input_size)
    torch.cuda.set_device(device)
    # cudnn.benchmark = True

    model.eval()
    model = model.cuda()
    input = []
    for i in range(1, n + 1):
        input_img = torch.randn(input_size, device=device)
        input.append(input_img)

    flops, _ = profile(model.to(device), (input,), verbose=False)
    flops = flops * 2 if operations == '* & +' else flops
    torch.cuda.empty_cache()

    params = sum([param.nelement() for param in model.parameters()])
    flops, params = clever_format([flops, params], "%.3f")

    for i in range(20):
        model(input)
    v_mem = round(torch.cuda.memory_allocated() / 1024 ** 2, 3)

    print('Total Params: %s' % params)
    print('Total FLOPS: %s' % flops)
    print('Total Memories: %sM' % v_mem)
    print('========= Calculate FPS=========')
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(input)
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    speed_time = elapsed_time / iteration * 1000
    fps = iteration / elapsed_time

    print('Elapsed Time: [%.6f s / %d iter]' % (elapsed_time, iteration))
    print('Speed Time: %.6f ms / iter   FPS: %.6f' % (speed_time, fps))

    return flops, params, speed_time, fps, str(v_mem) + 'M'


if __name__ == "__main__":
    # args
    input_shape = (3, 512, 512)  # channel, height, width
    model_type = ''
    n_imgs = 2
    operation = '*'  # write matrix multiplication and addition as only one operation
    # operation = '* & +'  # write matrix multiplication and addition as two operations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_chans = input_shape[0] if n_imgs == 1 else tuple([input_shape[0] for _ in range(n_imgs)])

    model = Build_MultiSenseSeg(8, in_chans=in_chans)

    model = model.to(device)
    # torch.save(model.state_dict(), "summary.pth")
    for i in model.children():
        print(i)
        print('==============================')

    with torch.no_grad():
        flops, params, speed_time, fps, v_mem = compute_speed(model, input_shape, device=0, n=n_imgs, operations=operation)
    table = PrettyTable()
    table.field_names = ['', 'Value']
    table.add_row(['Params', f'{params}'])
    table.add_row(['FLOPS', f'{flops}'])
    table.add_row(['Memories', f'{v_mem}'])
    table.add_row(['FPS', round(fps, 3)])
    table.add_row(['Speed', f'{round(speed_time, 3)}ms/iter'])
    print('\nShow:')
    print(table)
