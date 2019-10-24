/*! @file */

#ifndef PARTITION_HPP__
#define PARTITION_HPP__

#include "metis.h"
#include "linalgcpp.hpp"
#include <vector>
#include <cmath>

namespace linalgcpp
{

template<typename T>
linalgcpp::Vector<int> Partition(const linalgcpp::SparseMatrix<T> adjacency, int num_parts)
{
    //TODO: Error Checking
    // - adjacency should be square
    // - num_parts should be 1 or greater
    
    int error, objval;
    int ncon = 1;
    
    int options[METIS_NOPTIONS] = { };
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;
    
    int nodes = adjacency.Cols();
    std::vector<int> partitions_data(nodes);
    
    std::vector<int> indptr(adjacency.GetIndptr());
    std::vector<int> col_indices(adjacency.GetIndices());
    
    error = METIS_PartGraphKway(&nodes,
                                &ncon,
                                indptr.data(),
                                col_indices.data(),
                                NULL,
                                NULL,
                                NULL,
                                &num_parts,
                                NULL,
                                NULL,
                                NULL,
                                &objval,
                                partitions_data.data()
                                );
    
    linalgcpp::Vector<int> partitions(partitions_data);
    return partitions;
}

/* Right now this is assuming that there are no empty partitions / skipped integers
 *
 */
linalgcpp::SparseMatrix<double> GetWeightedInterpolator(const linalgcpp::Vector<int> partitions)
{
    int rows = partitions.size();
    int cols = linalgcpp::Max(partitions) + 1;
    
    std::vector<int> indptr(rows + 1);
    std::vector<double> data(rows);
    
    std::vector<int> counts(cols, 0);
    for(int partition : partitions) ++counts[partition];
    for(size_t node = 0; node < rows; ++node)
    {
        data[node] = 1.0 / std::sqrt(counts[partitions[node]]);
        indptr[node] = node;
    }
    indptr[rows] = rows;
    
    SparseMatrix<double> interpolation_matrix(indptr, partitions.data(), data, rows, cols);
    return interpolation_matrix;
}

linalgcpp::SparseMatrix<int> GetUnweightedInterpolator(const linalgcpp::Vector<int> partitions)
{
    int rows = partitions.size();
    int cols = linalgcpp::Max(partitions) + 1;
    
    std::vector<int> indptr(rows + 1);
    std::vector<int> data(rows, 1);
    
    for(size_t node = 0; node < rows; ++node)
    {
        indptr[node] = node;
    }
    indptr[rows] = rows;
    
    SparseMatrix<int> interpolation_matrix(indptr, partitions.data(), data, rows, cols);
    return interpolation_matrix;
}

} // namespace linalgcpp

#endif // PARTITION_HPP__
