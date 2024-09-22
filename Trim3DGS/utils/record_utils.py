import os

def write_mat(mat, size=20, step=20, acc='3'):
    out_str = ""
    for i in range(size):
        for j in range(size):
            try:
                out_str += ("{:." + acc + "f} ").format(mat[i * step, j* step])
            except:
                pass
        out_str += "\n"
    return out_str

def record_depth(path, mean_depth, median_depth, iteration):
    thres = 1.5
    save_root = path + "/depth_maps/"+ str(iteration)
    os.makedirs(save_root, exist_ok = True)
    with open (save_root + "/mean_depth.txt", "w") as f:
        f.write(write_mat(mean_depth[0]))
        f.write("\nmean: {:.3f}".format(mean_depth[mean_depth > thres].abs().mean().item()))
    with open (save_root + "/median_depth.txt", "w") as f:
        f.write(write_mat(median_depth[0]))
        f.write("\nmean: {:.3f}".format(median_depth[median_depth > thres].abs().mean().item()))
    with open (save_root + "/mean_median_diff.txt", "w") as f:
        f.write(str((mean_depth - median_depth).abs().mean().item()))

def record_rec(path, rec, iteration):
    save_root = path + "/recs/"+ str(iteration)
    os.makedirs(save_root, exist_ok = True)
    for i in range(3):
        with open (save_root + "/rec_{}.txt".format(i), "w") as f:
            f.write(write_mat(rec[i], acc='1'))
            f.write("\nmean: {:.3f}".format(rec[i].abs().mean().item()))