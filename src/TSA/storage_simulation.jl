using DataFrames
using Random
using Plots
using Distributions

# inverse demand funtion
function D_inv(X::Float64)::Float64
    return 1e10 * X ^ (-4)
end

function simulate_storage(
        D_inv::Function,             # Inverse demand function
        A_vec::Vector{Float64},      # Vector of production
        mean::Float64,               # Mean of X
        I::Int64 = 0,                # Initial storage
        C::Int64 = 30,               # Storage capacity
        r::Float64 = 0.02,           # Discount rate
        m_rev::Float64 = 0.0,        # Marginal revenue of storage
    )::Dict{String, Vector{Float64}}

    output, prices, storage, stocks = Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()

    for t in 1:length(A_vec)

        A::Float64 = A_vec[t]    # Get production at time t
        Q::Float64 = A
        P::Float64 = D_inv(Q)

        lhs = D_inv(mean) / (1 + r)
        rhs = P + m_rev * I

        # Find optimal storage
        while true
            # Store condition
            if rhs < lhs && I < C
                I += 1
                Q -= 1

                P = D_inv(Q)
                rhs = P + m_rev * I

                if rhs > lhs
                    break
                end

            # Deplete condition
            elseif rhs > lhs && I > 0
                I -= 1
                Q += 1

                P = D_inv(Q)
                rhs = P + m_rev * I

                if rhs < lhs
                    break
                end
            # Continue condition
            else
                break
            end
        end

        push!(output, A)
        push!(prices, P)
        push!(storage, I)
        push!(stocks, Q + I)
    end

    result_dict = Dict(
        "output" => output,
        "prices" => prices,
        "storage" => storage,
        "stocks" => stocks
    )

    return result_dict
end

simulation_configs = Dict(
    "No Storage" => Dict(
        "Initial Storage" => 0,
        "Storage Capacity" => 0,
        "Discount Rate" => 0.02,
        "Marginal Revenue" => 0.0,
        "color" => "gray",
        "linestyle" => "solid"
    ),
    "Storage" => Dict(
        "Initial Storage" => 0,
        "Storage Capacity" => 0,
        "Discount Rate" => 0.02,
        "Marginal Revenue" => 0.0,
        "color" => "red",
        "linestyle" => "dash"
    ),
    "Storage Expensive" => Dict(
        "Initial Storage" => 0,
        "Storage Capacity" => 0,
        "Discount Rate" => 0.02,
        "Marginal Revenue" => 0.0,
        "color" => "black",
        "linestyle" => "dot"
    ),
)

begin
    p1 = plot(title="Prices")
    p2 = plot(ylim=(-4, 34), title="Optimal storage", legend=false)
    p3 = plot(ylim=(75, 150), title="Stocks", legend=false)
    p4 = plot(title="Price volatility", legend=false)
    plots = [p1, p2, p3, p4]'

    mean::Float64 = 100.0
    A::Vector{Float64} = rand(Normal(mean, 5), 200)

    # crate a for loop which returns key value pairs
    for (i, (name, config)) in enumerate(simulation_configs)
        sim = simulate_storage(
            D_inv, A, mean,
            config["Initial Storage"],
            config["Storage Capacity"],
            config["Discount Rate"],
            config["Marginal Revenue"]
            )
        df = DataFrame(sim)
        df[!, :volatility] = [std(df.prices[max(1, i-10):i]) for i in 1:length(df.prices)]

        for (plot, col) in zip(plots, [:prices, :storage, :stocks, :volatility])
            plot!(
                plot, df[!, col],
                linewidth=0.9,
                color=config["color"],
                linestyle=config["linestyle"],
                label=name
                )
        end

    end

    plot(p1, p2, p3, p4, layout=(4,1), size=(600, 800), tight_layout=true)

end
