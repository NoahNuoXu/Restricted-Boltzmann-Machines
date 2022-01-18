import json
import sys
import collections

import matplotlib.pyplot as plt


def main():
    datasizes = [1, 5, 10, 50, 100]
    f1scores = []
    for datasize in datasizes:
        result_file = 'metrics_%dk.json' % datasize
        with open(result_file) as fin:
            results = json.loads(fin.read())
        f1score = 0
        for k, metrics in results.items():
            if f1score < metrics['f1score']:
                f1score = metrics['f1score']
        f1scores.append(f1score)
    plt.figure()
    plt.plot(datasizes, f1scores, marker='o')
    plt.xlabel('Data size (k)')
    plt.ylabel('F1 score')
    plt.title('Recommendation system performance by data size')

    for datasize in datasizes:
        result_file = 'metrics_%dk.json' % datasize
        with open(result_file) as fin:
            results = json.loads(fin.read())
        metric2k2values = collections.defaultdict(dict)
        for k, metrics in results.items():
            for metric in metrics:
                metric2k2values[metric][k] = metrics[metric]
        plt.figure()
        for metric in metric2k2values:
            k_value_list = sorted(metric2k2values[metric].items(), key=lambda x:int(x[0]))
            ks = [k for k, _ in k_value_list]
            values = [v for _, v in k_value_list]
            plt.plot(ks, values, marker='o', label=metric)
        plt.xlabel('Top K')
        plt.ylabel('Metric Score')
        plt.title('Recommendation system by %dk rating samples' % datasize)
        plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
