### Code based on file linear_regression.jl from the repository

########################
#Componentwise Boosting:
########################
"""
    calcunibeta(X::AbstractMatrix{<:AbstractFloat}, res::AbstractVector{<:AbstractFloat}, n::Int, p::Int)

Compute the univariate ordinary linear least squares (OLLS) estimator for each component 1:p.

# Arguments
- `X::AbstractMatrix{<:AbstractFloat}`: The input matrix of size (n x p) where n is the number of observations and p is the number of predictors.
- `res::AbstractVector{<:AbstractFloat}`: The response vector of length n.
- `n::Int`: The number of observations.
- `p::Int`: The number of predictors.

# Returns
- `unibeta::Vector`: A vector of length p consisting of the OLLS-estimators for each component.
- `denom::Vector`: A vector of length p consisting of the denominators for later re-scaling.
"""
function calcunibeta(X::AbstractMatrix{<:AbstractFloat}, res::AbstractVector{<:AbstractFloat}, n::Int, p::Int)
    unibeta = zeros(p)
    denom = zeros(p)

    idx_lst = findall(x->x!=0, vec(sum(X,dims=1)))
    #compute the univariate OLLS-estimator for each component from idx_lst:
    for j in idx_lst

       for i=1:n
          unibeta[j] += X[i, j]*res[i]
          denom[j] += X[i, j]*X[i, j]
       end

       unibeta[j] /= denom[j] 

    end

    #return a vector unibeta consisting of the OLLS-estimators and another vector, 
    #consisting of the denominators (for later re-scaling)
    return unibeta, denom 
end

"""
    compL2Boost!(β::AbstractVector{<:AbstractFloat}, X::AbstractMatrix{<:AbstractFloat}, y::AbstractVector{<:Number}, ϵ::Number, M::Int)

Perform componentwise L2-Boosting for regularized linear regression (i.e. variable selection).

This mutable function implements the componentwise L2-Boosting algorithm. It iteratively updates the initial coefficient vector `β` by re-fitting residuals via adding a re-scaled version of the currently optimal univariate Ordinary Least Squares (OLS) estimator in each iteration. The algorithm aims to minimize the L2 loss between the predicted values and the target vector `y`.

# Arguments
- `β::AbstractVector{<:AbstractFloat}`: Coefficient vector to be updated.
- `X::AbstractMatrix{<:AbstractFloat}`: Design matrix.
- `y::AbstractVector{<:Number}`: Response vector.
- `ϵ::Number`: Scalar value for re-scaling the selected OLS estimator.
- `M::Int`: Number of boosting iterations.
"""
function compL2Boost!(β::AbstractVector{<:AbstractFloat}, X::AbstractMatrix{<:AbstractFloat}, y::AbstractVector{<:Number}, ϵ::Number, M::Int)
    #determine the number of observations (e.g. cells) and the number of features (e.g. genes) in the training data:
    n, p = size(X)

    for step in 1:M

        #compute the residual as the difference of the target vector and the current fit:
        curmodel = X * β
        res = y .- curmodel

        #determine the p unique univariate OLLS estimators for fitting the residual vector res:
        unibeta, denom = calcunibeta(X, res, n, p) 

        #determine the optimal index of the univariate estimators resulting in the currently optimal fit:
        vals = collect(unibeta[j]^2 * denom[j] for j in 1:p)
        max_val = findmax(vals)[1]
        max_val_indices = findall(x -> x == max_val, vals)
        optindex = rand(max_val_indices)

        #update β by adding a re-scaled version of the selected OLLS-estimator, by a scalar value ϵ ∈ (0,1):
        β[optindex] += unibeta[optindex] * ϵ 

    end
end

"""
    get_beta_vectors(regression_data::Dict{Any, Any})

Compute the OLLS-estimator for each component of the design matrix for each communication.

# Arguments
- `regression_data::Dict{Any, Any}`: A dictionary containing the standardized design matrices and response vectors for each communication.

# Returns
- `beta_vectors::Dict{Any, Any}`: A dictionary containing the OLLS-estimators for each component of the design matrix for each communication.
"""
function get_beta_vectors(regression_data::Dict{Any, Any})
    ϵ = 0.2 #learning rate (step width) for the boosting
    M = 15 #number of boosting steps

    beta_vectors = Dict()
    for sel_communication in keys(regression_data)
        X = regression_data[sel_communication][1]
        y = regression_data[sel_communication][2]
        β = zeros(size(X, 2)) #start with a zero initialization of the coefficient vector
        compL2Boost!(β, X, y, ϵ, M)
        beta_vectors[sel_communication] = β
    end
    return beta_vectors
end