import numpy as np
import argparse


class Cell(object):
    def __init__(
            self, idx, best_predecessor
    ):
        self.idx = idx
        self.best_predecessor = best_predecessor
        self.cost = 0


class TimeStep(object):
    def __init__(
            self, index, n_cells, cells
    ):
        self.index = index
        self.n_cells = n_cells
        self.cells = cells


class MinPathFinder:

    def __init__(self):
        self.trellis = dict()
        self.time_steps = 0  # just an array with the cell config. eg. [3,4,4,3]
        self.cell_structure = []
        self.time_step_layers = []  # stores the cells relevant to all layers
        self.best_end_cell = None
        self.final_cost = np.power(10, 10)
        self.path_cells = []

    def parse_weight_file(self, weight_file):
        weight_lines = open(weight_file, 'r').readlines()
        time_step = 0

        for line in weight_lines:
            if line.startswith('======'):
                time_step += 1
                continue
            if self.encode_name(time_step, time_step + 1) not in self.trellis.keys():
                self.trellis[self.encode_name(time_step, time_step + 1)] = []

            self.trellis[self.encode_name(time_step, time_step + 1)].append(
                np.array([float(i) for i in line.replace('\n', '').split()]))

    def generate_cell_config(self):
        self.time_steps = len(list(self.trellis.keys())) + 1
        for time_step in range(self.time_steps):
            key = self.encode_name(time_step, time_step + 1)
            if key in self.trellis.keys():
                self.cell_structure.append(len(self.trellis[key][0]))
                self.trellis[key] = np.array(self.trellis[key])
        self.cell_structure.insert(0, len(self.trellis[self.encode_name(0, 1)]))

        for time_step in range(self.time_steps):
            n_cells = self.cell_structure[time_step]
            self.time_step_layers.append(TimeStep(time_step, n_cells, [
                Cell(self.encode_name(time_step+1, i+1), None) for i in range(n_cells)]))

    def viterbi_forward_pass(self):
        for time_step in range(self.time_steps - 1):
            for next_cell_index in range(len(self.time_step_layers[time_step + 1].cells)):
                next_cell = self.time_step_layers[time_step + 1].cells[next_cell_index]
                costs = np.array([cell.cost for cell in self.time_step_layers[time_step].cells])
                trellis_key = self.encode_name(time_step, time_step + 1)
                best_val = np.argmin(self.trellis[trellis_key][:, next_cell_index] + costs)
                next_cell.best_predecessor = self.time_step_layers[time_step].cells[best_val]
                next_cell.cost = np.min(self.trellis[trellis_key][:, next_cell_index] + costs)

        self.best_end_cell = self.time_step_layers[-1].cells[
            np.argmin([cell.cost for cell in self.time_step_layers[-1].cells])]
        self.final_cost = self.best_end_cell.cost
        print(f"best destination {self.best_end_cell.idx} ")
        print(f"min cost -> {self.final_cost}")

    def viterbi_backtrace(self, current_cell):
        if current_cell is None or current_cell.best_predecessor is None:
            return
        predecessor = current_cell.best_predecessor
        self.viterbi_backtrace(predecessor)
        self.path_cells.append(predecessor)
        print(f"{predecessor.idx}-> {predecessor.cost}\n")

    def write_output_to_file(self, out_file):
        out_file = open(out_file, 'w')
        out_file.write(f"format example\n")
        out_file.write(f"this notation {self.encode_name(3,4)} means 4(th) cell from 3(rd) layer\n")
        out_file.write('=================================\n')
        out_file.write(f"best destination {self.best_end_cell.idx} \n")
        out_file.write(f"min cost -> {self.final_cost}\n")
        out_file.write('path to follow:\n')

        for cell in self.path_cells:
            out_file.write(f"{cell.idx} [cost: {cell.cost} ]\n")
        out_file.close()

    def write_to_console(self):
        for cell in self.path_cells:
            print(f"{cell.idx} [cost: {cell.cost} ]\n")

    @staticmethod
    def encode_name(i, j):
        return f"{i} :{j}"


def main():
    parser = argparse.ArgumentParser(
        description='Employs Viterbi Algorithm to find the shortest path through a set of cells (cities for example)', )
    parser.add_argument('--weights_file', help='File path for the trellis')
    parser.add_argument('--output_file', help='file path to write the output')

    args = parser.parse_args()

    path_finder = MinPathFinder()
    path_finder.parse_weight_file(args.weights_file)
    path_finder.generate_cell_config()
    path_finder.viterbi_forward_pass()
    path_finder.viterbi_backtrace(path_finder.best_end_cell)
    path_finder.write_output_to_file(args.output_file)
    path_finder.write_to_console()


if __name__ == '__main__':
    main()
