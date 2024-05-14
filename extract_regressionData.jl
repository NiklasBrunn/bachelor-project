"""
    highest_interaction_matching(sel_group::Int, sel_communication::String, communications::Dict{Any, Any}, n_cells::Int, n_groups::Int)

Determine the cells with the highest interaction for each group in a given communication.

# Arguments
- `sel_group::Int`: The selected recepting group.
- `sel_communication::String`: The selected communication.
- `communications::Dict{Any, Any}`: A dictionary containing dataframes with interaction scores between all pairs of cells for each communication.
- `n_cells::Int`: The total number of cells in the dataset.
- `n_groups::Int`: The total number of groups in the dataset.

# Returns
- `highest_interaction_pairs::Vector{Int}`: A vector containing the cells with the highest interaction for each group in the selected communication.
"""
function highest_interaction_matching(sel_group::Int, sel_communication::String, communications::Dict{Any, Any}, n_cells::Int, n_groups::Int)
    interaction_df = communications[sel_communication]
    n_cells_per_group = n_cells ÷ n_groups

    highest_interaction_pairs = zeros(n_cells_per_group)
    for i in 1:n_cells_per_group
        start_row = ((sel_group - 1)*n_cells*n_cells_per_group)+(i-1)*n_cells+1
        end_row = start_row + n_cells - 1
        vals = interaction_df[start_row : end_row, :GeometricMean]
        max_val = findmax(vals)[1]
        max_val_indices = findall(x -> x == max_val, vals)
        idx = rand(max_val_indices)
        highest_interaction_row = start_row + idx - 1
        highest_interaction_pairs[i] = interaction_df[highest_interaction_row, :Sender]
    end
    highest_interaction_pairs = Int.(highest_interaction_pairs)
    return highest_interaction_pairs
end

"""
    standardize(X::AbstractArray; corrected_std::Bool=true, dims::Int=1)

Standardize the input data matrix by subtracting the mean and dividing by the standard deviation.

# Arguments
- `X::AbstractArray`: The input data matrix.
- `corrected_std::Bool=true`: A boolean indicating whether the corrected standard deviation should be used.
- `dims::Int=1`: The dimension along which the mean and standard deviation should be computed.

# Returns
- `X::AbstractArray`: The standardized data matrix.
"""
function standardize(X::AbstractArray; corrected_std::Bool=true, dims::Int=1)
    X = (X .- mean(X, dims=dims))./ std(X, corrected=corrected_std, dims=dims)

    # Replace NaN values with zeros:
    for i in 1:size(X)[2]
        if sum(isnan.(X[:, i])) > 0
            X[:, i] = zeros(size(X)[1])
        end
    end
    return X
end

"""
    get_X_y(dataset::Matrix{Float32}, sel_receptors::Vector{Any}, communications::Dict{Any, Any}, sel_group::Int, sel_communication::String, n_cells::Int, n_groups::Int)

Extract the design matrix X and the response vector y for a given group and communication.

# Arguments
- `dataset::Matrix{Float32}`: The dataset containing the expression values of all genes.
- `sel_receptors::Vector{Any}`: A vector containing the number of the selected receptors for each group.
- `communications::Dict{Any, Any}`: A dictionary containing dataframes with interaction scores between all pairs of cells for each communication.
- `sel_group::Int`: The selected recepting group.
- `sel_communication::String`: The selected communication.
- `n_cells::Int`: The total number of cells in the dataset.
- `n_groups::Int`: The total number of groups in the dataset.

# Returns
- `X::Matrix{Float32}`: The design matrix for given communication.
- `y::Vector{Float32}`: The response vector for given communication.
"""
function get_X_y(dataset::Matrix{Float32}, sel_receptors::Vector{Any}, communications::Dict{Any, Any}, sel_group::Int, sel_communication::String, n_cells::Int, n_groups::Int)
    # Define the data matrix X and the response vector y:
    sel_receptor = sel_receptors[sel_group]
    receptor_expression = dataset[:, sel_receptor]
    n_cells_per_group = n_cells ÷ n_groups
    start_cell = (sel_group - 1)*n_cells_per_group + 1
    end_cell = start_cell + n_cells_per_group - 1
    y = receptor_expression[start_cell:end_cell]

    highest_interaction_pairs = highest_interaction_matching(sel_group, sel_communication, communications, n_cells, n_groups)
    X = zeros(n_cells_per_group, size(dataset)[2])
    for i in 1:n_cells_per_group
        for j in 1: size(dataset)[2]
            X[i, j] = dataset[highest_interaction_pairs[i], j]
        end
    end
    return X, y
end

"""
    extract_regression_data(dataset::Matrix{Float32}, sel_receptors::Vector{Any}, communications::Dict{Any, Any}, n_cells::Int, n_groups::Int)

Extract the design matrices and response vectors for each communication.

# Arguments
- `dataset::Matrix{Float32}`: The dataset containing the expression values of all genes.
- `sel_receptors::Vector{Any}`: A vector containing the number of the selected receptors for each group.
- `communications::Dict{Any, Any}`: A dictionary containing dataframes with interaction scores between all pairs of cells for each communication.
- `n_cells::Int`: The total number of cells in the dataset.
- `n_groups::Int`: The total number of groups in the dataset.

# Returns
- `regression_data::Dict{Any, Any}`: A dictionary containing the standardized design matrices and response vectors for each communication.
"""
function extract_regression_data(dataset::Matrix{Float32}, sel_receptors::Vector{Any}, communications::Dict{Any, Any}, n_cells::Int, n_groups::Int)
    communications_keys = keys(communications)
    regression_data = Dict()
    for sel_communication in communications_keys
        sel_group = parse(Int, sel_communication[13])
        X, y = get_X_y(dataset, sel_receptors, communications, sel_group, sel_communication, n_cells, n_groups)
        regression_data[sel_communication] = (standardize(X), y)
    end
    return regression_data
end