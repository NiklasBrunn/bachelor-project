using Pkg

# Load necessary packages
#Pkg.add("RDatasets")
#Pkg.add("GLM")
#Pkg.add("DataFrames")
#Pkg.add("Plots")
#Pkg.add("Statistics")

using RDatasets #For loading a toy dataset (cars dataset)
using GLM #Package for linear regression (and generalized linear modeling)
using DataFrames #For data manipulation
using Plots #For plotting
using Statistics # For statistical operations

# Load the mtcars dataset
mtcars = dataset("datasets", "mtcars")

# Define the covariate matrix and the response vector:
X = Matrix(mtcars[!, 2:end-1]) # Data matrix, i.e. predictors (intercept, weight, horsepower)
#drop the first column of car model names ...
Y = mtcars[!, end] # Response vector (miles per gallon)

# (Optional:) Standardize the covariate matrix and the response vector for the prediction task:
X = (X .- mean(X, dims=1)) ./ std(X, dims=1) # Standardize the covariate matrix
Y = (Y .- mean(Y)) / std(Y) # Standardize the response vector

# Basic inspection of the dataset
println("First few rows of the mtcars dataset:")
println(first(mtcars, 5)) # Display the first 5 rows

println("\nSummary of the mtcars dataset:")
println(describe(mtcars)) # Summarize the dataset

# Plot the distribution of the 'wt' covariate
histogram(mtcars[!, :WT], bins=15, alpha=0.7, legend=false,
          xlabel="Weight (1000 lbs)", ylabel="Frequency",
          title="Distribution of Car Weights in mtcars Dataset"
)



###################
#Linear regression:
###################
# type: ?lm in the REPL for more information
lm_model = lm(X, Y)

# Display the regression results
println("\nLinear Regression Results:")
println(lm_model)
println("\nMean Squared Error LR:")
println(mean((Y-X*coef(lm_model)).^2))



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

    #compute the univariate OLLS-estimator for each component 1:p:
    for j = 1:p

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
        optindex = findmax(collect(unibeta[j]^2 * denom[j] for j in 1:p))[2]

        #update β by adding a re-scaled version of the selected OLLS-estimator, by a scalar value ϵ ∈ (0,1):
        β[optindex] += unibeta[optindex] * ϵ 

    end

end


ϵ = 0.2 #learning rate (step width) for the boosting 
M = 30 #number of boosting steps 
β = zeros(size(X, 2)) #start with a zero initialization of the coefficient vector

#Apply the componentwise L2-Boosting algorithm:
compL2Boost!(β, X, Y, ϵ, M) 

# Display the learned coefficients:
println(β)
println("\nMean Squared Error compBoost:")
println(mean((Y-X*β).^2))
heatmap(hcat(β, coef(lm_model)))