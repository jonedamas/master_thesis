using ARCHModels

function sent_index(scores::Vector{Float64}, β::Float64)::Vector{Float64}
    # create docstring
    """
    Calculate the sentiment index of a document using the sentiment scores of the sentences.
    The sentiment index of a sentence is calculated as the sum of the sentiment scores of the sentence and the sentiment scores of the previous sentences, weighted by an exponential decay function.

    Parameters
    ----------
    scores : Vector{Float64}
        The sentiment scores of the sentences in the document.

    Returns
    -------
    sent_index : Vector{Float64}
        The sentiment index of the document.
    """
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
