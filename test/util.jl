# util.jl — 精确（暴力态矢）参照
using LinearAlgebra

"按 QuantumCircuits 门矩阵约定（`positions[1]` = 最高位）把门作用到态矢上。"
function qc_apply_vec(v::AbstractVector{T}, m::AbstractMatrix, positions) where {T<:Number}
    n = round(Int, log2(length(v)))
    k = length(positions)
    dim = 1 << n
    positions = collect(Int, positions)
    (size(m) == (1 << k, 1 << k)) || error("gate size mismatch")
    mbit = [k - j for j in 1:k]          # matrix bit（QC 约定：positions[j] 的位）
    vbit = [p - 1 for p in positions]    # 态矢位（bit(site)=site-1）
    out = zeros(complex(float(T)), dim)
    for i in 0:(dim-1)
        ai = 0
        for t in 1:k
            ai |= ((i >> vbit[t]) & 1) << mbit[t]
        end
        acc = zero(complex(float(T)))
        for jj in 0:(dim-1)
            same = true
            for b in 0:(n-1)
                b in vbit && continue              # 门位自由变化
                (((jj >> b) & 1) != ((i >> b) & 1)) && (same = false; break)
            end
            same || continue
            aj = 0
            for t in 1:k
                aj |= ((jj >> vbit[t]) & 1) << mbit[t]
            end
            acc += m[ai+1, aj+1] * v[jj+1]
        end
        out[i+1] = acc
    end
    return out
end

"线路精确参照：依次作用 `(mat, positions)`。"
function exact_simulate(n::Int, gates)
    v = zeros(ComplexF64, 1 << n)
    v[1] = 1
    for (m, ps) in gates
        v = qc_apply_vec(v, m, ps)
    end
    return v
end

"态保真度 |⟨a|b⟩|²/(‖a‖²‖b‖²)。"
fid(a::AbstractVector, b::AbstractVector) = abs(dot(a, b))^2 / (norm(a)^2 * norm(b)^2)

# 常用门矩阵（QC 约定：positions[1] 为最高位 → 用于两比特时手动给出位序）
const _H = ComplexF64[1 1; 1 -1] ./ sqrt(2)
const _S = ComplexF64[1 0; 0 im]
const _T = ComplexF64[1 0; 0 exp(im * pi / 4)]
const _X = ComplexF64[0 1; 1 0]
const _Z = ComplexF64[1 0; 0 -1]
const _CX = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]   # 控制=最高位(q[1])
const _CZ = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1]
