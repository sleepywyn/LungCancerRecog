import numpy as np
import simple_reader as sr

npz_path = "./prepretrain_stage2"
big_size = 216
small_size = 36
output_folder = "./lc_cubics"
test_csv_path = "./data/stage2_sample_submission.csv"

def get_patch_from_list(lung_img, coords, cub_size):
	shape = lung_img.shape
	output = []
	lung_img = lung_img + 1024
	for i in range(len(coords)):
		patch = lung_img[coords[i][0] - cub_size / 2: coords[i][0] + cub_size / 2,
				coords[i][1] - cub_size / 2: coords[i][1] + cub_size / 2,
				coords[i][2] - cub_size / 2: coords[i][2] + cub_size / 2]
		output.append(patch)
	return output


def get_coords(big_size, small_size, conv=False):
    mid = small_size / 2
    if conv == False:
        array = [0, 0, 0]
        cubeMid = [array]*big_size
        i=0
        for x in range(mid, big_size-mid+1,small_size):
            for y in range(mid,big_size-mid+1,small_size):
                for z in range(mid,big_size-mid+1,small_size):
                    cubeMid[i]= [x,y,z]
                    i=i+1
    else:
        array = [0, 0, 0]
        cubeMid = [array]*(big_size-mid*2+1)*(big_size-mid*2+1)*(big_size-mid*2+1)
        i=0
        for x in range(mid, big_size-mid+1):
            for y in range(mid,big_size-mid+1):
                for z in range(mid,big_size-mid+1):
                    cubeMid[i]= [x,y,z]
                    i=i+1
    return cubeMid


def gen_cubics():
	coords = get_coords(big_size, small_size)
	print(coords)
	with open(test_csv_path, 'rb') as f:
		lines = f.readlines()[1:]
		for line in lines:
			p_id = line.split(',')[0]
			lung_img = sr.load_npz(npz_path + "/" + p_id + ".npz")
			print("INFO: Before cutting, Lung image has shape: " + str(lung_img.shape))
			
			if (lung_img.shape[0] - big_size) % 2 == 1:
				cut_slice1 = (lung_img.shape[0] - big_size) / 2
				cut_slice2 = (lung_img.shape[0] - big_size) / 2 + 1
			else:
				cut_slice1 = cut_slice2 = (lung_img.shape[0] - big_size) / 2
			
			cut_slice = (lung_img.shape[1] - big_size) / 2
			lung_img = lung_img[cut_slice1:-cut_slice2, cut_slice:-cut_slice, cut_slice:-cut_slice]
			print("INFO: After cutting, Lung image has shape: " + str(lung_img.shape))
			
			testX = get_patch_from_list(lung_img, coords, small_size)
			
			count = 0
			for cubic in testX:
                		np.savez_compressed(output_folder + "/" + p_id + "_" + str(count), np.asarray(cubic))
                		count += 1


if __name__ == '__main__':
	gen_cubics()
