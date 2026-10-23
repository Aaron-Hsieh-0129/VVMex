#include "io/hdf5/CompressedHdf5Writer.hpp"
#include <cstdio>
#include <cstring>

using VVM::IO::CompressedHdf5Writer;

template<class T>
void field(CompressedHdf5Writer& writer, const std::string& name,
    const std::vector<size_t>& shape, int rank, int ranks) {
    const size_t begin = shape[0] * rank / ranks;
    const size_t end = shape[0] * (rank + 1) / ranks;
    auto count = shape;
    count[0] = end - begin;
    std::vector<size_t> start(shape.size(), 0);
    start[0] = begin;
    size_t plane = 1;
    for (size_t d = 1; d < shape.size(); ++d) plane *= shape[d];
    std::vector<T> data((end-begin)*plane);
    for (size_t i = 0; i < data.size(); ++i) data[i] = T((begin*plane+i)%301)*T(0.125);
    writer.write(name, shape, start, count, data.empty() ? nullptr : data.data());
}

template<class T>
void inspect(hid_t file, const char* name, size_t size, hid_t type, bool compressed) {
    hid_t d = H5Dopen2(file, name, H5P_DEFAULT);
    if (d < 0) throw std::runtime_error("Missing dataset");
    std::vector<T> actual(size), expected(size);
    for (size_t i = 0; i < size; ++i) expected[i] = T(i%301)*T(0.125);
    if (H5Dread(d, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, actual.data()) < 0 ||
        std::memcmp(actual.data(), expected.data(), size*sizeof(T)) != 0)
        throw std::runtime_error("Readback differs");
    hid_t props = H5Dget_create_plist(d);
    if (H5Pget_nfilters(props) != (compressed ? 3 : 0))
        throw std::runtime_error("Filter count differs");
    if (compressed) {
        const H5Z_filter_t wanted[] = {H5Z_FILTER_SHUFFLE,H5Z_FILTER_DEFLATE,H5Z_FILTER_FLETCHER32};
        for (unsigned i=0;i<3;++i) {
            unsigned flags = 0;size_t n=0;
            if (H5Pget_filter2(props,i,&flags,&n,nullptr,0,nullptr,nullptr)!=wanted[i])
                throw std::runtime_error("Filter order differs");
        }
    }
    H5Pclose(props);H5Dclose(d);
}

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    int rank=0,ranks=1;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    try {
        if (argc!=2) throw std::runtime_error("usage: test_compressed_hdf5 OUTPUT.h5");
        VVM::IO::Hdf5CompressionOptions options;options.enabled=true;options.minimum_bytes=1048576;
        {
            CompressedHdf5Writer writer(argv[1],MPI_COMM_WORLD,options);
            writer.attribute("units",std::string("test"));
            writer.attribute<double>("earth_radius_m",{6371220.0});
            const int64_t step=INT64_C(9007199254740993);
            writer.write("model_step",{},{},{},&step);
            field<float>(writer,"rain",{513,513},rank,ranks);
            // With four ranks two have no cells; chunks are also shared in rain.
            field<double>(writer,"u",{2,129,513},rank,ranks);
            field<double>(writer,"coordinates/x",{513},rank,ranks);
            writer.close();
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank==0) {
            hid_t file=H5Fopen(argv[1],H5F_ACC_RDONLY,H5P_DEFAULT);
            inspect<float>(file,"/Step0/rain",513*513,H5T_NATIVE_FLOAT,true);
            inspect<double>(file,"/Step0/u",2*129*513,H5T_NATIVE_DOUBLE,true);
            inspect<double>(file,"/Step0/coordinates/x",513,H5T_NATIVE_DOUBLE,false);
            hid_t d=H5Dopen2(file,"/Step0/model_step",H5P_DEFAULT);
            int64_t step=0;
            if(H5Dread(d,H5T_NATIVE_INT64,H5S_ALL,H5S_ALL,H5P_DEFAULT,&step)<0 || step!=INT64_C(9007199254740993))
                throw std::runtime_error("Integer clock lost precision");
            H5Dclose(d);H5Fclose(file);
            std::printf("PASS: %d ranks, bitwise readback, filters, empty ranks, exact clock\n",ranks);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    } catch(const std::exception& e) {
        std::fprintf(stderr,"rank %d: %s\n",rank,e.what());MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();return 0;
}
