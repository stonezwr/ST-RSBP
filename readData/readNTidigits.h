#ifndef _READ_NTIDIGITS_DATA_H_
#define _READ_NTIDIGITS_DATA_H_

#include "../common/cuMatrix.h"
#include "../common/cuMatrixVector.h"
#include "../common/util.h"
#include "../common/MemoryMonitor.h"
#include <string>
#include <vector>

//* read each speech file
void read_each_NTidigits(const std::string& filename, cuMatrixVector<bool>& x, int nrows, int ncols);

//* read each speech from the dump file
void read_each_NTidigits_dump(const std::string& filename, cuMatrixVector<bool>& x, int nrows, int ncols);

//* read the dumped input of CPU as a spike time matrix
void read_dumped_input_inside_NTidigits(const std::string& filename, cuMatrixVector<bool>& x, int nrows, int ncols);

//* read training data and lables
int readNTidigits(cuMatrixVector<bool> &x,
	cuMatrix<int>* &y, 
	std::string path,
	int number_of_speeches,
	int input_neurons,
    int end_time,
    int CLS);

//* read the labels
int readNTidigitsLabel(const std::vector<int>& labels, cuMatrix<int>* &mat);

//* read the samples and label (encoded in the directory)
int readNTidigits(std::string path, cuMatrixVector<bool>& x, std::vector<int>& labels, int num, int input_neurons, int end_time, int CLS);

#endif
