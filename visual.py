import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import csv
def load_graph(file_path,id_filename=None):
    G=nx.Graph()
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(file_path, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3 or row[0] == '' or row[1] == '' or row[2] == '':
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                G.add_edge(id_dict[i], id_dict[j], weight=distance)
        return G
    else:
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3 or row[0] == '' or row[1] == '' or row[2] == '':
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                G.add_edge(i, j, weight=distance)
        return G

def flow_graph(graph_signal_matrix_filename,All=True):
    if All:
        data = np.load(graph_signal_matrix_filename)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #绘制第一个探测器的流量变化
        data=data['data'][:100,:,0:1]
        T,N,C=data.shape
        T = np.arange(1, T + 1)
        for i in range(N):
            C=data[:,i,0].reshape(-1)
            ax.plot(T, C, zs=i, zdir='x')
        ax.set_xlabel('num')
        ax.set_ylabel('time')
        ax.set_zlabel('flow')
        ax.legend()
        plt.show()
    else:
        data = np.load(graph_signal_matrix_filename)
        data = data['data'][:100, 0:1, 0:1]
        T = np.arange(1, 101)
        C = data.reshape(-1)
        plt.plot(T, C)
        plt.xlabel('time')
        plt.ylabel('flow')
        plt.title("the first detector's flow")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    file_path="data/PEMS03/PEMS03.csv"
    id_filename="data/PEMS03/PEMS03.txt"
    graph_signal_matrix_filename="data/PEMS03/PEMS03.npz"
    flow_graph(graph_signal_matrix_filename,All=True)

    G=load_graph(file_path,id_filename)
    pos = nx.random_layout(G)  # 选择布局方式
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=100, edge_color='black', linewidths=0.2,
            font_size=2)
    # 显示图
    plt.show()
