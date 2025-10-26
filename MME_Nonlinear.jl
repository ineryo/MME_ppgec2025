module MME_Nonlinear
# Mini-biblioteca de métodos para zeros de funções e sistemas não lineares
# Julia 1.8+ | sem dependências externas
# Autor: você :) | foco didático e reuso

export bisection, false_position, secant, newton1d, fixed_point, newton_nd,
       numdiff, numjac, bracket

using LinearAlgebra

# ---------- Utilidades numéricas ----------

"""
    numdiff(f, x; h = √eps()*max(1, abs(x)))

Derivada numérica (diferença central) de `f` em `x`.
"""
function numdiff(f::F, x::T; h = √eps()*max(one(T), abs(x))) where {F,T<:Real}
    fxp = f(x + h)
    fxm = f(x - h)
    return (fxp - fxm) / (2h)
end

"""
    numjac(F, x; hscale = √eps())

Jacobiana numérica por diferenças centrais para `F::VectorFunction`.
`x` é vetor. `hscale` controla o passo relativo por componente.
"""
function numjac(F::F, x::AbstractVector; hscale = √eps()) where {F}
    n = length(x)
    Fx = F(x)
    m = length(Fx)
    J = Matrix{Float64}(undef, m, n)
    for j in 1:n
        h = hscale*max(1.0, abs(x[j]))
        ej = zeros(eltype(x), n); ej[j] = 1.0
        J[:, j] = (F(x .+ h.*ej) - F(x .- h.*ej)) ./ (2h)
    end
    return J
end

# Critérios genéricos de parada (relativo no passo e no resíduo)
const _TOLX_DEFAULT = 1e-12
const _TOLF_DEFAULT = 1e-12

_stop_step(x, xnew; tolx=_TOLX_DEFAULT) = norm(xnew .- x) ≤ tolx*max(1.0, norm(xnew))
_stop_resid(fx; tolf=_TOLF_DEFAULT)     = abs(fx) ≤ tolf
_stop_resid_vec(Fx; tolf=_TOLF_DEFAULT) = norm(Fx) ≤ tolf

# Resultado padrão
_result(x; fx=nothing, it=0, ok=false, why="") = (
    x = x, fx = fx, iters = it, converged = ok, reason = why
)

# ---------- Funções 1D ----------

"""
    bracket(f, x0; step=1.0, growth=1.6, maxiter=50)

Tenta encontrar [a,b] com mudança de sinal partindo de `x0`.
Retorna (a,b, fa, fb) se encontrar; lança erro caso contrário.
"""
function bracket(f, x0; step=1.0, growth=1.6, maxiter=50)
    a = x0; fa = f(a)
    b = x0 + step; fb = f(b)
    it = 0
    while fa*fb > 0 && it < maxiter
        it += 1
        step *= growth
        a = x0 - step; fa = f(a)
        b = x0 + step; fb = f(b)
    end
    fa*fb > 0 && error("bracket: não achou mudança de sinal a partir de x0=$x0.")
    return (a,b,fa,fb)
end

"""
    bisection(f, a, b; tolx=1e-12, tolf=1e-12, maxiter=10_000)

Bisseção robusta em [a,b] com f(a)·f(b) ≤ 0.
"""
function bisection(f, a::Real, b::Real; tolx=_TOLX_DEFAULT, tolf=_TOLF_DEFAULT, maxiter=10_000)
    a > b && ((a,b) = (b,a))
    fa, fb = f(a), f(b)
    fa*fb > 0 && error("bisseção exige f(a)·f(b) ≤ 0.")
    it = 0
    while (b - a) > tolx && it < maxiter
        it += 1
        c  = (a + b)/2
        fc = f(c)
        _stop_resid(fc; tolf=tolf) && return _result(c; fx=fc, it=it, ok=true, why="|f|≤tolf")
        if fa*fc ≤ 0
            b, fb = c, fc
        else
            a, fa = c, fc
        end
    end
    x = (a + b)/2
    return _result(x; fx=f(x), it=it, ok=(b-a)≤tolx, why="intervalo≤tolx")
end

"""
    false_position(f, a, b; tolx=1e-12, tolf=1e-12, maxiter=10_000, method=:illinois)

Regula Falsi (puro) ou variante Illinois (mais estável). `method` ∈ {:classic, :illinois}.
"""
function false_position(f, a::Real, b::Real; tolx=_TOLX_DEFAULT, tolf=_TOLF_DEFAULT,
                        maxiter=10_000, method::Symbol=:illinois)
    a > b && ((a,b) = (b,a))
    fa, fb = f(a), f(b)
    fa*fb > 0 && error("false_position exige f(a)·f(b) ≤ 0.")
    it = 0
    wa, wb = 1.0, 1.0  # pesos para Illinois
    x, fx = a, fa
    while it < maxiter
        it += 1
        # interpolação linear ponderada
        x = (a*wb*fb - b*wa*fa) / (wb*fb - wa*fa)
        fx = f(x)
        _stop_resid(fx; tolf=tolf) && return _result(x; fx=fx, it=it, ok=true, why="|f|≤tolf")
        _stop_step(x, a; tolx=tolx) && _stop_step(x, b; tolx=tolx) &&
            return _result(x; fx=fx, it=it, ok=true, why="passo≤tolx")
        if fa*fx < 0
            b, fb = x, fx
            wb = 1.0
            method === :illinois && (wa *= 0.5)  # enfraquece o lado que não troca
        else
            a, fa = x, fx
            wa = 1.0
            method === :illinois && (wb *= 0.5)
        end
    end
    return _result(x; fx=fx, it=it, ok=false, why="maxiter")
end

"""
    secant(f, x0, x1; tolx=1e-12, tolf=1e-12, maxiter=10_000)

Método da secante 1D (sem derivações).
"""
function secant(f, x0::Real, x1::Real; tolx=_TOLX_DEFAULT, tolf=_TOLF_DEFAULT, maxiter=10_000)
    f0, f1 = f(x0), f(x1)
    it = 0
    while it < maxiter
        it += 1
        den = (f1 - f0)
        den == 0 && return _result(x1; fx=f1, it=it, ok=false, why="denominador=0")
        x2 = x1 - f1*(x1 - x0)/den
        f2 = f(x2)
        _stop_resid(f2; tolf=tolf) && return _result(x2; fx=f2, it=it, ok=true, why="|f|≤tolf")
        _stop_step(x1, x2; tolx=tolx) && return _result(x2; fx=f2, it=it, ok=true, why="passo≤tolx")
        x0, x1, f0, f1 = x1, x2, f1, f2
    end
    return _result(x1; fx=f1, it=it, ok=false, why="maxiter")
end

"""
    newton1d(f, x0; df=nothing, tolx=1e-12, tolf=1e-12, maxiter=10_000, safe=false)

Newton 1D. Se `df==nothing`, usa derivada numérica central.
Se `safe=true`, faz backtracking (Armijo) no passo.
"""
function newton1d(f, x0::Real; df=nothing, tolx=_TOLX_DEFAULT, tolf=_TOLF_DEFAULT,
                  maxiter=10_000, safe=false)
    x = float(x0)
    fx = f(x)
    it = 0
    while it < maxiter
        it += 1
        _stop_resid(fx; tolf=tolf) && return _result(x; fx=fx, it=it, ok=true, why="|f|≤tolf")
        dfx = df === nothing ? numdiff(f, x) : df(x)
        dfx == 0 && return _result(x; fx=fx, it=it, ok=false, why="f'(x)=0")
        step = -fx/dfx
        λ = 1.0
        if safe
            # Armijo simples: reduz λ até melhorar f (em norma escalar)
            f0 = abs(fx)
            while λ > 1e-6
                xt = x + λ*step
                ft = f(xt)
                if abs(ft) ≤ (1 - 1e-4*λ)*f0
                    x, fx = xt, ft
                    break
                end
                λ *= 0.5
            end
            λ ≤ 1e-6 && (x += step; fx = f(x)) # aceita mesmo assim
        else
            x += step
            fx = f(x)
        end
        _stop_step(x - step, x; tolx=tolx) && return _result(x; fx=fx, it=it, ok=true, why="passo≤tolx")
    end
    return _result(x; fx=fx, it=it, ok=false, why="maxiter")
end

"""
    fixed_point(g, x0; tolx=1e-12, maxiter=10_000, λ=1.0, aitken=false)

Iteração de ponto-fixo x_{k+1} = (1-λ)x_k + λ g(x_k).
Se `aitken=true`, aplica aceleração Δ² de Aitken a cada passo.
"""
function fixed_point(g, x0::Real; tolx=_TOLX_DEFAULT, maxiter=10_000, λ=1.0, aitken=false)
    x = float(x0)
    it = 0
    xprev = x
    while it < maxiter
        it += 1
        x1 = (1-λ)*x + λ*g(x)
        if aitken
            # Aitken Δ²: x̂ = x - (Δx)^2 / (x2 - 2x1 + x)
            x2 = (1-λ)*x1 + λ*g(x1)
            den = (x2 - 2x1 + x)
            if den != 0
                xhat = x - (x1 - x)^2 / den
                xprev, x = x, xhat
            else
                xprev, x = x, x2
            end
        else
            xprev, x = x, x1
        end
        _stop_step(xprev, x; tolx=tolx) && return _result(x; fx=(x - g(x)), it=it, ok=true, why="passo≤tolx")
    end
    return _result(x; fx=(x - g(x)), it=it, ok=false, why="maxiter")
end

# ---------- Sistemas não lineares (R^n → R^n) ----------

"""
    newton_nd(F, x0; J=nothing, tolx=1e-12, tolf=1e-12, maxiter=200, linesearch=true)

Newton–Raphson para sistemas. Se `J==nothing`, usa Jacobiana numérica.
`linesearch=true` aplica backtracking (Armijo) no passo.
Retorna NamedTuple com `x`, `fx::Vector`, etc.
"""
function newton_nd(F, x0::AbstractVector; J=nothing, tolx=_TOLX_DEFAULT,
                   tolf=_TOLF_DEFAULT, maxiter=200, linesearch=true)
    x = collect(float.(x0))
    Fx = F(x)
    it = 0
    while it < maxiter
        it += 1
        _stop_resid_vec(Fx; tolf=tolf) && return _result(x; fx=Fx, it=it, ok=true, why="‖F‖≤tolf")
        Jx = J === nothing ? numjac(F, x) : J(x)
        # resolve J Δ = -F com fallback de mínimos quadrados se mal-condicionado
        Δ = - (Jx \ Fx)
        # linha de busca (Armijo)
        λ = 1.0
        xnew = x .+ λ .* Δ
        Fnew = F(xnew)
        if linesearch
            f0 = norm(Fx)
            while norm(Fnew) > (1 - 1e-4*λ)*f0 && λ > 1e-6
                λ *= 0.5
                xnew = x .+ λ .* Δ
                Fnew = F(xnew)
            end
        end
        _stop_step(x, xnew; tolx=tolx) && return _result(xnew; fx=Fnew, it=it, ok=true, why="passo≤tolx")
        x, Fx = xnew, Fnew
    end
    return _result(x; fx=Fx, it=it, ok=false, why="maxiter")
end

# ---------- Exemplos rápidos (rodam só se arquivo for executado diretamente) ----------
if abspath(PROGRAM_FILE) == @__FILE__
    f(x) = cos(x) - x
    println("Bisseção em [0,1]: ", bisection(f, 0.0, 1.0))
    println("Secante (0,1):      ", secant(f, 0.0, 1.0))
    println("Newton1D:           ", newton1d(f, 0.5; df=x->-sin(x)-1))

    # Sistema: F([x,y]) = [x^2 + y^2 - 1, x - y]
    F(v) = [v[1]^2 + v[2]^2 - 1.0,
            v[1] - v[2]]
    println("Newton nD:          ", newton_nd(F, [0.7, 0.2]))
end

end # module
