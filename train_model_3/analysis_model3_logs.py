import tkinter as tk
from tkinter.filedialog import *
import numpy as np
import matplotlib.pyplot as plt


class GUI():
    def __init__(self, init_window):
        self.init_window = init_window  # 窗口名
        self.create_window()  # 创建窗口控件函数

        self.filename = tk.StringVar()  # 读取文件路径名
        self.data = np.array([])  # 读取的数据

        self.para_dict = {'train_precision': 1, 'train_recall': 2, 'train_f1': 3, 'train_loss': 4,
                          'test_precision': 5, 'test_recall': 6, 'test_f1': 7, 'test_loss':8 }

        self.parameter_label = tk.StringVar()  # 所选参量名
        self.parameter_data = np.array([])      #所选参量的数值列表

        self.max_epoch=tk.StringVar()
        self.max_precision=tk.StringVar()
        self.max_recall=tk.StringVar()
        self.max_f1=tk.StringVar()
        self.train_or_test=''


        self.parameter_list = [0, 0, 0, 0, 0, 0, 0, 0]

    # 读取文件的按键响应
    def selectFile(self, ):
        def read_train_logs(file_path):
            f = open(file_path, 'r', encoding='utf-8')
            lines = f.readlines()
            res = []
            for line in lines[1:]:
                line = line.split()
                res.append(line)
            return np.array(res, dtype=np.float)

        filepath = askopenfilename()  # 选择打开什么文件，返回文件名
        self.filename.set(filepath)  # 设置变量filename的值
        self.data = read_train_logs(filepath)
        if self.data.any():

            tk.Label(self.init_window, text='读取成功    ' + str(os.path.basename(filepath))).grid(row=1, column=3, padx=5,
                                                                                               pady=5)
        else:
            tk.Label(self.init_window, text='读取失败').grid(row=1, column=3, padx=5, pady=5)
        self.parameter_label.set('')
        self.max_epoch.set('')
        self.max_precision.set('')
        self.max_recall.set('')
        self.max_f1.set('')

    # 创建窗口上控件
    def create_window(self):
        # 显示打开文件
        tk.Label(self.init_window, text='选择文件').grid(row=1, column=0, padx=5, pady=5)
        tk.Button(self.init_window, text='打开文件', command=self.selectFile).grid(row=1, column=2, padx=5, pady=5)

        # 显示要查看的参数
        tk.Label(self.init_window, text='参数统计部分', height=2).grid(row=2, column=0, padx=5, pady=5)
        tk.Label(self.init_window, text='选择的依据参数：', height=2).grid(row=3, column=0, padx=5, pady=5)

        #显示统计结果
        tk.Label(self.init_window, text='precision', height=2).grid(row=5, column=1, padx=5,)
        tk.Label(self.init_window, text='recall', height=2).grid(row=5, column=2, padx=5,)
        tk.Label(self.init_window, text='f1', height=2).grid(row=5, column=3, padx=5,)

        # 显示结果
        tk.Label(self.init_window, text='结果为：', height=2).grid(row=6, column=0, padx=5, pady=5)
        tk.Label(self.init_window, text='轮次：', height=2).grid(row=7, column=0, padx=5, pady=5)

        # 显示画图部分选项
        tk.Label(self.init_window, text='画曲线图部分', height=2).grid(row=8, column=0, padx=5, pady=5)


    # <editor-fold desc="选择参量的按键响应">
    def result(self):
        self.max_epoch.set(int(np.argwhere(self.parameter_data==max(self.parameter_data))[-1]))
        if self.train_or_test=='train':
            self.max_precision.set(self.data[int(self.max_epoch.get())][1]*100)
            self.max_recall.set(self.data[int(self.max_epoch.get())][2]*100)
            self.max_f1.set(self.data[int(self.max_epoch.get())][3]*100)
        else:
            self.max_precision.set(self.data[int(self.max_epoch.get())][5]*100)
            self.max_recall.set(self.data[int(self.max_epoch.get())][6]*100)
            self.max_f1.set(self.data[int(self.max_epoch.get())][7]*100)

    def para_train_precison(self):
        try:
            self.parameter_data = self.data.T[1]
            self.parameter_label.set('train_precison')
            self.train_or_test='train'
            self.result()
        except:
            tk.Label(self.init_window, text='请先选择文件，读取数据', height=2).grid(row=1, column=3, padx=5, pady=5)

    def para_train_recall(self):
        try:
            self.parameter_data = self.data.T[2]
            self.parameter_label.set('train_recall')
            self.train_or_test = 'train'
            self.result()
        except:
            tk.Label(self.init_window, text='请先选择文件，读取数据', height=2).grid(row=1, column=3, padx=5, pady=5)

    def para_train_f1(self):
        try:
            self.parameter_data = self.data.T[3]
            self.parameter_label.set('train_f1')
            self.train_or_test = 'train'
            self.result()
        except:
            tk.Label(self.init_window, text='请先选择文件，读取数据', height=2).grid(row=1, column=3, padx=5, pady=5)

    def para_test_precision(self):
        try:
            self.parameter_data = self.data.T[5]
            self.parameter_label.set('test_precision')
            self.train_or_test = 'test'
            self.result()
        except:
            tk.Label(self.init_window, text='请先选择文件，读取数据', height=2).grid(row=1, column=3, padx=5, pady=5)

    def para_test_recall(self):
        try:
            self.parameter_data = self.data.T[6]
            self.parameter_label.set('test_recall')
            self.train_or_test = 'test'
            self.result()
        except:
            tk.Label(self.init_window, text='请先选择文件，读取数据', height=2).grid(row=1, column=3, padx=5, pady=5)

    def para_test_f1(self):
        try:
            self.parameter_data = self.data.T[7]
            self.parameter_label.set('test_f1')
            self.train_or_test = 'test'
            self.result()
        except:
            tk.Label(self.init_window, text='请先选择文件，读取数据', height=2).grid(row=1, column=3, padx=5, pady=5)

    # </editor-fold>

    #画图的按键响应
    def plot(self):
        try:
            x = self.data[2]
            mat_data = {}
            for i, label in zip(self.parameter_list, self.para_dict):
                if i:
                    mat_data[label] = self.data.T[self.para_dict[label]]

            for label in mat_data:
                if len(mat_data)==1:
                    plt.ylabel(label)
                    plt.plot(mat_data[label])
                else:
                    plt.ylabel(label.replace('train_','').replace('test_',''))
                    plt.plot(mat_data[label], label=label)

            plt.xlabel('epoch')
            plt.ylabel('values')
            plt.xlabel('epoch')
            plt.legend()
            plt.show()
        except:
            tk.Label(self.init_window, text='请先选择文件，读取数据', height=2).grid(row=1, column=3, padx=5, pady=5)

    def set_init_window(self):
        # 显示文件路径
        tk.Entry(self.init_window, textvariable=self.filename, ).grid(row=1, column=1, padx=5, pady=5)

        # <editor-fold desc="显示想要统计的参量">
        tk.Entry(self.init_window, textvariable=self.parameter_label).grid(row=3, column=1, padx=5, pady=5)
        tk.Button(self.init_window, text='train_precison', height=2, width=20, command=self.para_train_precison).grid(row=4,
                                                                                                            column=0,
                                                                                                            padx=5,
                                                                                                            pady=5)
        tk.Button(self.init_window, text='train_recall', height=2, width=20, command=self.para_train_recall).grid(row=4,
                                                                                                            column=1,
                                                                                                            padx=5,
                                                                                                            pady=5)
        tk.Button(self.init_window, text='train_f1', height=2, width=20, command=self.para_train_f1).grid(row=4,
                                                                                                              column=2,
                                                                                                              padx=5,
                                                                                                              pady=5)
        tk.Button(self.init_window, text='test_precision', height=2, width=20, command=self.para_test_precision).grid(row=4,
                                                                                                          column=3,
                                                                                                          padx=5,
                                                                                                          pady=5)
        tk.Button(self.init_window, text='test_recall', height=2, width=20, command=self.para_test_recall).grid(row=4,
                                                                                                          column=4,
                                                                                                          padx=5,
                                                                                                          pady=5)
        tk.Button(self.init_window, text='test_f1', height=2, width=20, command=self.para_test_f1).grid(row=4,
                                                                                                            column=5,
                                                                                                            padx=5,
                                                                                                            pady=5)
        # </editor-fold>

        # <editor-fold desc="显示统计函数的结果">
        tk.Entry(self.init_window, textvariable=self.max_precision).grid(row=6, column=1, padx=5, pady=5)
        tk.Entry(self.init_window, textvariable=self.max_epoch).grid(row=7, column=1, padx=5, pady=5)
        tk.Entry(self.init_window, textvariable=self.max_recall).grid(row=6, column=2, padx=5, pady=5)
        tk.Entry(self.init_window, textvariable=self.max_f1).grid(row=6, column=3, padx=5, pady=5)
        # </editor-fold>


        # 画图部分
        # <editor-fold desc="画曲线图">
        def check_list():
            self.parameter_list = [0, 0, 0, 0, 0, 0, 0, 0]
            if var1.get():
                self.parameter_list[0] = 1
            if var2.get():
                self.parameter_list[1] = 1
            if var3.get():
                self.parameter_list[2] = 1
            if var4.get():
                self.parameter_list[3] = 1
            if var5.get():
                self.parameter_list[4] = 1
            if var6.get():
                self.parameter_list[5] = 1
            if var7.get():
                self.parameter_list[6] = 1
            if var8.get():
                self.parameter_list[7] = 1

        var1 = tk.IntVar()
        var2 = tk.IntVar()
        var3 = tk.IntVar()
        var4 = tk.IntVar()
        var5 = tk.IntVar()
        var6 = tk.IntVar()
        var7 = tk.IntVar()
        var8 = tk.IntVar()
        tk.Checkbutton(self.init_window, text='train_precison', variable=var1, onvalue=1, offvalue=0,
                       command=check_list).grid(row=9, column=0, pady=5)
        tk.Checkbutton(self.init_window, text='train_recall', variable=var2, onvalue=1, offvalue=0,
                       command=check_list).grid(row=9, column=1)
        tk.Checkbutton(self.init_window, text='train_f1', variable=var3, onvalue=1, offvalue=0,
                       command=check_list).grid(row=9, column=2)
        tk.Checkbutton(self.init_window, text='train_loss', variable=var4, onvalue=1, offvalue=0,
                       command=check_list).grid(row=9, column=3)
        tk.Checkbutton(self.init_window, text='test_precision', variable=var5, onvalue=1, offvalue=0,
                       command=check_list).grid(row=10, column=0)
        tk.Checkbutton(self.init_window, text='test_recall', variable=var6, onvalue=1, offvalue=0,
                       command=check_list).grid(row=10, column=1)
        tk.Checkbutton(self.init_window, text='test_f1', variable=var7, onvalue=1, offvalue=0,
                       command=check_list).grid(row=10, column=2)
        tk.Checkbutton(self.init_window, text='test_loss', variable=var8, onvalue=1, offvalue=0,
                       command=check_list).grid(row=10, column=3)

        tk.Button(self.init_window, text='画图', height=2, width=20, command=self.plot).grid(row=11, column=0, pady=5)
        # </editor-fold>



        return


if __name__ == '__main__':
    window = tk.Tk()
    window.title('analysis_model3_logs')
    window.geometry('1000x600')
    my_window = GUI(window)
    my_window.set_init_window()
    window.mainloop()