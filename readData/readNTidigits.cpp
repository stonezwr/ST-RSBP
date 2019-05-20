#include "readNTidigits.h"
#include <sstream>
#include <fstream>
#include <vector>
#include <assert.h>
#include <sys/stat.h>
#include <dirent.h>
#include <cuda_runtime_api.h>


//* recursively find the files
void file_finder_NTidigits(const std::string& path, cuMatrixVector<bool>& x, std::vector<int>& labels, int cur_label, int& sample_count, int num_of_samples, int end_time, int input_neurons, int CLS)
{
    DIR *dir;
    struct dirent *ent; 
    struct stat st;

    dir = opendir(path.c_str());
    while((ent = readdir(dir)) != NULL){
        if(sample_count >= num_of_samples)  return;

        std::string file_name = ent->d_name;
        std::string full_file_name = path + "/" + file_name;
        if(file_name[0] == '.') continue;
    
        if(stat(full_file_name.c_str(), &st) == -1) continue;

        bool is_directory = (st.st_mode & S_IFDIR) != 0;
        if(file_name.length() <= 2){
            cur_label = atoi(file_name.c_str());
            assert(cur_label >= 0 && cur_label < CLS);
        }
        
        if(is_directory){
            assert(cur_label >= 0 && cur_label < CLS);
            file_finder_NTidigits(full_file_name, x, labels, cur_label, sample_count, num_of_samples, end_time, input_neurons, CLS); 
        }
        else{
            // this is indeed the data file:
            string suffix = ".dat";
            assert(file_name.length() >= suffix.length() && file_name.substr(file_name.length() - suffix.length()) == suffix);
            read_each_NTidigits(full_file_name, x, end_time, input_neurons);

            labels.push_back(cur_label);
            sample_count++;
            printf("read %2d%%", 100 * sample_count / num_of_samples);
        }
        printf("\b\b\b\b\b\b\b\b");
    }

}


//* read the train data and label of speeches at the same time
int readNTidigits(
        std::string path, 
        cuMatrixVector<bool>& x,
        std::vector<int>& labels,
        int num,
        int input_neurons,
        int end_time,
        int CLS)
{
    //* read the data from the path
    struct stat sb;
    if(stat(path.c_str(), &sb) != 0){
        std::cout<<"The given path: "<<path<<" does not exist!"<<std::endl;
        exit(EXIT_FAILURE);
    } 

    if(path[path.length() - 1] == '/')  path = path.substr(0, path.length() - 1);
    //* recursively read the samples in the directory
    int sample_count = 0;
    file_finder_NTidigits(path, x, labels, -1, sample_count, num, end_time, input_neurons, CLS);
    assert(x.size() == num);
    assert(x.size() == labels.size());

    return x.size();
}

//* read each sample of Speech dataset
void read_each_NTidigits(const std::string& filename, cuMatrixVector<bool>& x, int nrows, int ncols)
{
    std::ifstream f_in(filename.c_str());
    if(!f_in.is_open()){
        std::cout<<"Cannot open the file: "<<filename<<std::endl;
        exit(EXIT_FAILURE);
    }
    // get all the analog values of the speeches
	std::vector<vector<int> > * sp_time = new std::vector<vector<int> >(ncols, vector<int>());
    std::string times;
	int input_channel = 0;
    while(getline(f_in, times)){
        std::istringstream iss(times);
        int t;
		int tmp_t=0;
		assert(input_channel<= ncols);
        while(iss>>t){
			if(t<tmp_t){
				t=tmp_t+1;
			}
			(*sp_time)[input_channel].push_back(t);
			tmp_t=t;
		}
		input_channel++;
    }
    f_in.close();
    if(input_channel != ncols){
        std::cout<<"The number of channels in the raw speech file: "<<input_channel
                 <<" does not match the number of input neuron: "<<ncols<<std::endl;
        exit(EXIT_FAILURE);
    }
    cuMatrix<bool>* tpmat = new cuMatrix<bool>(nrows, ncols, 1, sp_time);
    tpmat->freeCudaMem(); 
    x.push_back(tpmat);
}


//* read each speech from the dump file of the CPU simulator
void read_each_NTidigits_dump(const std::string& filename, cuMatrixVector<bool>& x, int nrows, int ncols)
{
    std::ifstream f_in(filename.c_str());
    if(!f_in.is_open()){
        std::cout<<"Cannot open the file: "<<filename<<std::endl;
        exit(EXIT_FAILURE);
    }
    cuMatrix<bool>* tpmat = new cuMatrix<bool>(nrows, ncols, 1);
    tpmat->freeCudaMem();
    
    int index, spike_time;
    f_in>>index>>spike_time; // get rid of -1   -1 at the beginning
    while(f_in>>index>>spike_time){
        if(index == -1 && spike_time == -1) break; // only read one iteration of speech
        assert(index < ncols);
        if(spike_time >= nrows) continue;
        tpmat->set(spike_time, index, 0, true);
    }   
    f_in.close();
    x.push_back(tpmat);
}

//* read the dumped input of CPU as a spike time matrix
void read_dumped_input_inside_NTidigits(const std::string& filename, cuMatrixVector<bool>& x, int nrows, int ncols)
{
    std::ifstream f_in(filename.c_str());
    if(!f_in.is_open()){
        std::cout<<"Cannot open the file: "<<filename<<std::endl;
        exit(EXIT_FAILURE);
    }
    vector<vector<int> > * sp_time = new vector<vector<int> >(ncols, vector<int>()); 
    int index, spike_time;
    f_in>>index>>spike_time; // get rid of -1   -1 at the beginning
    while(f_in>>index>>spike_time){
        if(index == -1 && spike_time == -1) break; // only read one iteration of speech
        assert(index < ncols);
        if(spike_time >= nrows) continue;
        assert(spike_time > ((*sp_time)[index].empty() ? -1 : (*sp_time)[index].back()));

        (*sp_time)[index].push_back(spike_time);
    }   
    f_in.close();
    cuMatrix<bool>* tpmat = new cuMatrix<bool>(nrows, ncols, 1, sp_time);
    tpmat->freeCudaMem();
    x.push_back(tpmat);
}

//* read the label
int readNTidigitsLabel(const std::vector<int>& labels, cuMatrix<int>* &mat){
    for(int i = 0; i < labels.size(); ++i){
        mat->set(i, 0, 0, labels[i]);
    }
    mat->toGpu(); 
    return labels.size();
}

//* read trainning data and lables
int readNTidigits(
        cuMatrixVector<bool>& x,
        cuMatrix<int>*& y, 
        std::string path,
        int number_of_speeches,
        int input_neurons,
        int end_time,
        int CLS)
{

    std::vector<int> labels;
    int len = readNTidigits(path, x, labels, number_of_speeches, input_neurons, end_time, CLS);
    //* read speech label into cuMatrix
    y = new cuMatrix<int>(len, 1, 1);
    int t = readNTidigitsLabel(labels, y);
    return t;
}
