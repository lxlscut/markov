import cv2
import numpy as np


def add_noise(origin_image):
    A = np.copy(origin_image)
    for i in range(origin_image.shape[0]):
        for j in range(origin_image.shape[1]):
            r = np.random.rand()
            if r<0.1:
                # print(A[i,j])
                A[i][j] = -A[i][j]
                # print(A[i][j])
    return A


def compute_log_prob_helper(noisy_image_copy, i, j):
    try:
        return noisy_image_copy[i][j]
    except IndexError:
        return 0


def compute_log_prob(noisy_image, noisy_image_copy, i, j, w_e, w_s, y_val):
    result = w_e*noisy_image[i][j]*y_val
    result += w_s*y_val*compute_log_prob_helper(noisy_image_copy,i-1,j)
    result += w_s*y_val*compute_log_prob_helper(noisy_image_copy,i+1,j)
    result += w_s*y_val*compute_log_prob_helper(noisy_image_copy,i,j-1)
    result += w_s*y_val*compute_log_prob_helper(noisy_image_copy,i,j+1)
    return result


def denoised_image(noisy_image,w_e,w_s):
    m,n = np.shape(noisy_image)[:2]
    noisy_image_copy = np.copy(noisy_image)
    max_iter = 10*m*n
    for iter in range(max_iter):
        i = np.random.randint(m)
        j = np.random.randint(n)
        log_p_neg = compute_log_prob(noisy_image,noisy_image_copy,i,j,w_e,w_s,-1)
        log_p_pos = compute_log_prob(noisy_image,noisy_image_copy,i,j,w_e,w_s,1)
        # print(log_p_pos,log_p_neg)
        if log_p_neg>log_p_pos:
            noisy_image_copy[i][j] = -1
        else:
            noisy_image_copy[i][j] = 1
        if iter%100000 == 0:
            print('Complete', iter, 'iterations out of',max_iter)
    return noisy_image_copy


if __name__ == '__main__':
    input = cv2.imread("input.png",0)
    input = cv2.threshold(input,100,2,cv2.THRESH_BINARY)[1]
    input = np.array(input,dtype=np.int)-1
    # input = input/255
    noise_img = add_noise(origin_image=input)
    denoised_image=denoised_image(noise_img,8,10)
    noise_img = noise_img.astype(np.uint8)
    cv2.imshow("noise_img",noise_img*255)
    denoised_image = denoised_image.astype(np.uint8)
    cv2.imshow("denoised_img",denoised_image*255)
    # cv2.imshow("input",input*255)
    cv2.waitKey(0)
