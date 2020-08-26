import numpy as np
import cv2 as cv
import copy


class MRF():
    def __init__(self, img, max_iter=100, num_clusters=5, init_func=None, beta=8e-4):
        self.max_iter = max_iter
        self.kernels = np.zeros(shape=(8, 3, 3))
        self.beta = beta
        #todo 标签的数量
        self.num_clusters = num_clusters
        #todo 获取其八邻域的像素值
        for i in range(9):
            if i < 4:
                self.kernels[i, i // 3, i % 3] = 1
            elif i > 4:
                self.kernels[i - 1, i // 3, i % 3] = 1
        #todo 灰度图归一化后的值
        self.img = img
        if init_func is None:
            self.labels = np.random.randint(low=1, high=num_clusters + 1, size=img.shape, dtype=np.uint8)

    def __call__(self):
        img = self.img.reshape((-1,))
        # todo 最外层的大循环
        for iter in range(self.max_iter):
            # todo p1的作用，大小为标签数量*图片的大小
            p1 = np.zeros(shape=(self.num_clusters, self.img.shape[0] * self.img.shape[1]))
            # todo 对每一个标签进行循环
            for cluster_idx in range(self.num_clusters):
                #todo
                temp = np.zeros(shape=(self.img.shape))
                #todo 每一个像素周围的8个标签的值,看周围的八个像素提供的标签参考
                for i in range(8):
                    res = cv.filter2D(self.labels, -1, self.kernels[i, :, :])
                    temp[(res == (cluster_idx + 1))] -= self.beta
                    temp[(res != (cluster_idx + 1))] += self.beta
                temp = np.exp(-temp)
                #todo 标签信息被存放在p中
                p1[cluster_idx, :] = temp.reshape((-1,))
            # todo 转换成概率？？？
            p1 = p1 / np.sum(p1)
            # todo 如果是0也将其赋予一定的概率
            p1[p1 == 0] = 1e-3
            # todo 这里的mu为均值，sigma为方差
            mu = np.zeros(shape=(self.num_clusters,))
            sigma = np.zeros(shape=(self.num_clusters,))
            for i in range(self.num_clusters):
                # mu[i] = np.mean(self.img[self.labels == (i+1)])
                # todo 因为labels为1,2,而原来的值的范围为【0,2】
                data = self.img[self.labels == (i + 1)]
                if np.sum(data) > 0:
                    mu[i] = np.mean(data)
                    sigma[i] = np.var(data)

                else:
                    mu[i] = 0
                    sigma[i] = 1
                # print(sigma[i])
            # sigma[sigma == 0] = 1e-3
            #todo p2的作用是计算p(fs|ws)
            p2 = np.zeros(shape=(self.num_clusters, self.img.shape[0] * self.img.shape[1]))
            # todo 对于每一个像素
            for i in range(self.img.shape[0] * self.img.shape[1]):
                # todo 判断其标签
                for j in range(self.num_clusters):
                    # print(sigma[j])
                    # todo 这里计算的P(fs|Ws),先验概率,i guess so
                    p2[j, i] = -np.log(np.sqrt(2 * np.pi) * sigma[j]) - (img[i] - mu[j]) ** 2 / 2 / sigma[j];

            self.labels = np.argmax(np.log(p1) + p2, axis=0) + 1
            self.labels = np.reshape(self.labels, self.img.shape).astype(np.uint8)
            print("-----------start-----------")
            print(p1)
            print("-" * 20)
            print(p2)
            print("----------end------------")
            # print("iter {} over!".format(iter))
            # self.show()
            # print(self.labels)

    def show(self):
        h, w = self.img.shape
        show_img = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        show_img[self.labels == 1, :] = (0, 255, 255)
        show_img[self.labels == 2, :] = (220, 20, 60)
        show_img[self.labels == 3, :] = (65, 105, 225)
        show_img[self.labels == 4, :] = (50, 205, 50)
        # img = self.labels / (self.num_clusters) * 255

        cv.imshow("res", show_img)
        cv.waitKey(0)


if __name__ == "__main__":
    img = cv.imread("woman.jpg")

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img / 255.
    # img = np.random.rand(64,64)
    # img = cv.resize(img,(256,256))
    mrf = MRF(img=img, max_iter=20, num_clusters=4)
    mrf()
    mrf.show()
    # print(mrf.kernels)