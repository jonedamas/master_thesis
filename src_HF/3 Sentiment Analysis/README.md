# Sentiment Analysis

## sent_utils.py functions

### aggregate_score()

The function `aggregate_score()` is used to aggregate the sentiment scores of the news articles. The function takes in the sentiment scores of the news articles and the number of news articles and returns the average sentiment score.

$$
SV_t = \frac{1}{N_t}\sum_{i=1}^{n} PV_{it}
$$


### SI_bai()

The function `SI_bai()` is used to calculate the sentiment index of the news articles. The function takes in the sentiment scores of the news articles and the number of news articles and returns the sentiment index.

$$
SI_t = SV_t+ \sum_{i=1}^{t-1} SV_i\cdot e^{-\frac{t-1}{\beta}}
$$

The function computations is written in **Julia** with the following code:

```julia
function SI_bai(scores::Vector{Float64}, β::Float64)::Vector{Float64}

    sent_index = Vector{Float64}(undef, length(scores))

    for t in 1:length(scores)
        SV = scores[t]
        for i in 1:(t - 1)
            SV += scores[i] * exp(-(t-i)/β)
        end
        sent_index[t] = SV
    end
    return sent_index
end
```
