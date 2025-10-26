{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "\n# ES5100 — Métodos Matemáticos para Engenharia  \n## Lista 4 — Zeros de Funções e Sistemas Não Lineares (Notebook em Julia)\n\n**Objetivo.** Usar a biblioteca criada (módulo `MME`) para resolver os exercícios da **Lista 4**:\n- Métodos: bisseção, ponto fixo (Picard), Newton, secante e Newton-Raphson para sistemas.\n- Comparar velocidades de convergência quando solicitado e apresentar resultados com **6 algarismos significativos**.\n\n> Referências da disciplina e da lista: material de “Zero de funções e sistemas não-lineares” e enunciados da Lista 4.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "\n### Preparação do ambiente\n\nA célula abaixo tenta carregar a biblioteca `MME`.  \nSe o pacote não estiver disponível, definimos **funções fallback** mínimas compatíveis com a interface usada neste notebook.\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "execution_count": null,
      "outputs": [],
      "source": [
        "\n# Ambiente básico\nusing Printf\nusing LinearAlgebra\nconst TOL_DEFAULT = 1e-12\nconst MAXIT_DEFAULT = 100\n\n# Tenta usar a biblioteca criada\nconst _has_MME = try\n    @eval using MME  # ajuste aqui caso seu módulo tenha outro nome\n    true\ncatch\n    @warn \"Pacote `MME` não encontrado. Usando funções fallback locais.\"\n    false\nend\n\n# -----------------------------\n# Fallbacks (somente se MME não estiver disponível)\n# -----------------------------\nif !_has_MME\n    struct IterLog{T}\n        k::Int\n        x::T\n        fx::T\n        err::T\n    end\n\n    function newton_fallback(f, df, x0; tol=TOL_DEFAULT, maxiter=MAXIT_DEFAULT, log=false)\n        x = x0\n        logs = IterLog{Float64}[]\n        for k in 1:maxiter\n            fx = f(x)\n            dfx = df(x)\n            Δ = fx/dfx\n            xnew = x - Δ\n            err = abs(xnew - x)\n            if log\n                push!(logs, IterLog(k, xnew, f(xnew), err))\n            end\n            x = xnew\n            if err < tol\n                return x, logs\n            end\n        end\n        return x, logs\n    end\n\n    function bisection_fallback(f, a, b; tol=TOL_DEFAULT, maxiter=MAXIT_DEFAULT, log=false)\n        fa, fb = f(a), f(b)\n        if fa*fb > 0\n            error(\"Bisseção requer f(a)*f(b) < 0\")\n        end\n        logs = IterLog{Float64}[]\n        left, right = a, b\n        for k in 1:maxiter\n            mid = (left + right)/2\n            fm = f(mid)\n            err = (right - left)/2\n            if log; push!(logs, IterLog(k, mid, fm, err)); end\n            if abs(fm) < tol || err < tol\n                return mid, logs\n            end\n            if fa*fm < 0\n                right = mid\n                fb = fm\n            else\n                left = mid\n                fa = fm\n            end\n        end\n        return (left+right)/2, logs\n    end\n\n    function fixed_point_fallback(g, x0; tol=TOL_DEFAULT, maxiter=MAXIT_DEFAULT, log=false)\n        x = x0\n        logs = IterLog{Float64}[]\n        for k in 1:maxiter\n            xnew = g(x)\n            err = abs(xnew - x)\n            if log; push!(logs, IterLog(k, xnew, NaN, err)); end\n            x = xnew\n            if err < tol\n                return x, logs\n            end\n        end\n        return x, logs\n    end\n\n    function secant_fallback(f, x0, x1; tol=TOL_DEFAULT, maxiter=MAXIT_DEFAULT, log=false)\n        logs = IterLog{Float64}[]\n        x_prev, x = x0, x1\n        f_prev, f_curr = f(x_prev), f(x)\n        for k in 1:maxiter\n            denom = f_curr - f_prev\n            if denom == 0\n                error(\"Denominador zero na secante\")\n            end\n            xnew = x - f_curr*(x - x_prev)/denom\n            err = abs(xnew - x)\n            if log; push!(logs, IterLog(k, xnew, f(xnew), err)); end\n            x_prev, x = x, xnew\n            f_prev, f_curr = f_curr, f(x)\n            if err < tol\n                return x, logs\n            end\n        end\n        return x, logs\n    end\n\n    function newton_system_fallback(F, J, x0::AbstractVector; tol=TOL_DEFAULT, maxiter=MAXIT_DEFAULT, log=false)\n        x = copy(x0)\n        for k in 1:maxiter\n            Fx = F(x)\n            if norm(Fx, Inf) < tol\n                return x, k\n            end\n            Jx = J(x)\n            Δ = -Jx \\ Fx\n            x .= x .+ Δ\n            if norm(Δ, Inf) < tol\n                return x, k\n            end\n        end\n        return x, maxiter\n    end\n\n    # Wrappers com nomes da \"biblioteca\"\n    const MME = Module(:MME)\n    @eval MME begin\n        export newton, bisection, fixed_point, secant, newton_system\n    end\n    MME.newton = newton_fallback\n    MME.bisection = bisection_fallback\n    MME.fixed_point = fixed_point_fallback\n    MME.secant = secant_fallback\n    MME.newton_system = newton_system_fallback\nend\n\n# Aliases locais para usar abaixo (se pacote existir, usam as versões do pacote; caso contrário, os fallbacks)\nconst newton        = MME.newton\nconst bisection     = MME.bisection\nconst fixed_point   = MME.fixed_point\nconst secant        = MME.secant\nconst newton_system = MME.newton_system\n\nprintln(\"Ambiente pronto. Usando biblioteca MME = \", _has_MME)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "\n## Exercício 4–1 — Newton vs. Bisseção vs. Ponto Fixo (Picard)\n\nResolver a raiz positiva com 6 algarismos significativos de  \n\\[\nf(x) = e^{-x} - 2\\sqrt{x}, \\quad x\\in[0,1].\n\\]\n1) Explique por que **não** é conveniente usar \\(x^{(0)}=0\\) (derivada envolve \\(1/\\sqrt{x}\\)).  \n2) Compare a velocidade de convergência de Newton e Bisseção.  \n3) Refaça por um **ponto fixo asseguradamente convergente**: escolha \\(g(x) = \\frac{1}{4}e^{-2x}\\), pois \\(|g'(x)|=\\frac{1}{2}e^{-2x}\\le \\tfrac{1}{2}<1\\) em \\([0,\\infty)\\), garantindo contração.\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "execution_count": null,
      "outputs": [],
      "source": [
        "\n# Definição do problema\nf(x)  = exp(-x) - 2*sqrt(x)\ndf(x) = -exp(-x) - (2)*(1/(2*sqrt(x)))  # -e^{-x} - 1/sqrt(x)\n\n# Observação: x0 = 0 é ruim pois df(0) é singular (1/sqrt(x) → ∞).\nprintln(\"Observação: x0=0 é inadequado para Newton pois df(x) tem termo 1/sqrt(x).\")\n\n# 1) Newton\nx0_N = 0.5\nrootN, logN = newton(f, df, x0_N; tol=1e-13, maxiter=100, log=true)\n@printf(\"Newton: raiz ≈ %.10f (|f|=%.3e) em %d iterações\\n\", rootN, abs(f(rootN)), length(logN))\n\n# 2) Bisseção (intervalo [0,1], f(0)= -2*0 = 0? Cuidado: f(0)= exp(0) - 0 = 1 > 0; f(1)= e^{-1}-2 <0)\na, b = 0.0, 1.0\nrootB, logB = bisection(f, a, b; tol=1e-12, maxiter=100, log=true)\n@printf(\"Bisseção: raiz ≈ %.10f (|f|=%.3e) em %d iterações\\n\", rootB, abs(f(rootB)), length(logB))\n\n# 3) Ponto fixo com contração garantida: x = g(x) = 1/4 * exp(-2x)\ng(x) = 0.25*exp(-2x)\nx0_G = 0.3\nrootG, logG = fixed_point(g, x0_G; tol=1e-13, maxiter=200, log=true)\n@printf(\"Ponto Fixo: raiz ≈ %.10f (|f|=%.3e) em %d iterações\\n\", rootG, abs(f(rootG)), length(logG))\n\n# Comparativo simples de iterações\n@printf(\"\\nComparativo de iterações: Newton=%d, Bisseção=%d, Ponto Fixo=%d\\n\",\n        length(logN), length(logB), length(logG))\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "\n## Exercício 4–2 — Fator de atrito em tubo liso (escoamento turbulento)\n\nResolver para \\(f>0\\) a equação\n\\[\n\\sqrt{f} = f\\,[\\,1.74\\,\\ln(\\mathrm{Re}\\sqrt{f}) - 0.4\\,], \\quad \\mathrm{Re}=5000,\n\\]\nvia **Newton**, com chute inicial pela equação de **Blasius** \\( f = 0.316\\,\\mathrm{Re}^{-0.25}\\).  \nDepois, verificar a (in)adequação de aplicar **substituições sucessivas** isolando \\(f\\) no lado direito.\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "execution_count": null,
      "outputs": [],
      "source": [
        "\nRe = 5000.0\nf0_blasius = 0.316 * Re^(-0.25)\n@printf(\"Chute de Blasius: f0 ≈ %.8f\\n\", f0_blasius)\n\nφ(f) = sqrt(f) - f*(1.74*log(Re*sqrt(f)) - 0.4)\nφ′(f) = (1/(2*sqrt(f))) - 1.74*(log(Re*sqrt(f)) + 0.5) + 0.4\n\nfN, logF = newton(φ, φ′, f0_blasius; tol=1e-14, maxiter=100, log=true)\n@printf(\"Newton: f ≈ %.10f, φ(f)=%.3e, it=%d\\n\", fN, φ(fN), length(logF))\n\n# Substituições sucessivas diretas (isolar f no lado direito):\n# f = sqrt(f) / (1.74*ln(Re*sqrt(f)) - 0.4)  ==> define g(f)\nfunction g_bad(f)\n    denom = 1.74*log(Re*sqrt(f)) - 0.4\n    return sqrt(f)/denom\nend\n\n# Verifica derivada numérica de g no ponto solução para julgar contração (|g'(f*)|<1?)\nfunction numderiv(h, x; eps=1e-7)\n    (h(x+eps) - h(x-eps)) / (2eps)\nend\n\ngprime = numderiv(g_bad, fN)\n@printf(\"g'(f*) ≈ %.4f  => %s\\n\", gprime, (abs(gprime)<1 ? \"contração (converge)\" : \"não contração (pode divergir)\"))\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "\n## Exercício 4–3 — Equação de estado de Van der Waals\n\n\\[\n\\big(P+\\frac{a}{V^2}\\big)\\,(V-b) = RT \\;\\;\\Longleftrightarrow\\;\\; P(V;T)=\\frac{RT}{V-b}-\\frac{a}{V^2}.\n\\]\n\nPara cada gás da tabela, computar \\(T_c=\\frac{8a}{27Rb}\\).  \n- Para \\(T>T_c\\): mostrar que há única solução para qualquer \\(P\\).  \n- Para \\(T<T_c\\): calcular a faixa \\([P_{\\min}(T),P_{\\max}(T)]\\) na qual existem **três soluções** \\(V_1<V_2<V_3\\) (obtida pelos extremos locais com \\(dP/dV=0\\)), e resolver um caso ilustrativo dentro e fora dessa faixa.\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "execution_count": null,
      "outputs": [],
      "source": [
        "\n# Constante dos gases\nR = 0.082054  # L atm mol^-1 K^-1\n\n# Tabela (a [L^2 atm mol^-2], b [L mol^-1])\ngases = [\n    (\"CO2\", 3.592,   0.04267),\n    (\"Anilina dimetílica\", 37.490, 0.19700),\n    (\"He\",   0.03412, 0.02370),\n    (\"NO\",   1.340,   0.02789),\n]\n\nTc(a,b) = 8a/(27R*b)\n\n# P(V;T) e dP/dV\nP_of_V(V, T, a, b) = R*T/(V - b) - a/(V^2)\ndPdV(V, T, a, b)   = -R*T/(V - b)^2 + 2a/(V^3)\n\n# Encontrar extremos (raízes de dP/dV) para T<Tc\nfunction vdW_extrema(T, a, b; Vmin=b*1.0001, Vmax=200b, nscan=2000)\n    # varredura para achar mudanças de sinal em dP/dV\n    Vs = range(Vmin, Vmax; length=nscan)\n    sgn_prev = sign(dPdV(first(Vs), T, a, b))\n    roots = Float64[]\n    vprev = first(Vs)\n    for v in Iterators.drop(Vs,1)\n        s = sign(dPdV(v, T, a, b))\n        if sgn_prev == 0\n            push!(roots, vprev)\n        elseif s != sgn_prev\n            # bisseção local\n            left, right = vprev, v\n            for _ in 1:80\n                mid = (left+right)/2\n                if dPdV(left, T, a, b)*dPdV(mid, T, a, b) <= 0\n                    right = mid\n                else\n                    left = mid\n                end\n            end\n            push!(roots, (left+right)/2)\n        end\n        sgn_prev = s\n        vprev = v\n    end\n    sort(roots)\nend\n\n# Resolver P(V)=Pfixo encontrando até 3 raízes (bisseção em subintervalos com mudança de sinal)\nfunction solve_V_for_P(Pfix, T, a, b; Vmin=b*1.0001, Vmax=200b, nscan=5000, tol=1e-10)\n    F(V) = P_of_V(V,T,a,b) - Pfix\n    Vs = range(Vmin, Vmax; length=nscan)\n    vals = [F(v) for v in Vs]\n    roots = Float64[]\n    for i in 1:length(Vs)-1\n        if vals[i]*vals[i+1] < 0\n            left, right = Vs[i], Vs[i+1]\n            for _ in 1:80\n                mid = (left+right)/2\n                if F(left)*F(mid) <= 0\n                    right = mid\n                else\n                    left = mid\n                end\n            end\n            push!(roots, (left+right)/2)\n        end\n    end\n    sort(roots)\nend\n\nfor (name, a, b) in gases\n    Tc_ab = Tc(a,b)\n    @printf(\"\\nGás: %-20s  a=%.5f  b=%.5f  =>  Tc ≈ %.3f K\\n\", name, a, b, Tc_ab)\n\n    # Caso T > Tc\n    T_high = 1.10*Tc_ab\n    # P(V) é monotônica: uma única solução para qualquer P (ilustramos para P=1 atm)\n    Vroot_high = solve_V_for_P(1.0, T_high, a, b)\n    @printf(\"  T=1.10 Tc: para P=1 atm, solução única V ≈ %g L/mol\\n\", Vroot_high[1])\n\n    # Caso T < Tc\n    T_low = 0.90*Tc_ab\n    ext = vdW_extrema(T_low, a, b)\n    if length(ext) == 2\n        V1e, V2e = ext[1], ext[2]\n        Pmax = P_of_V(V1e, T_low, a, b)  # máximo local\n        Pmin = P_of_V(V2e, T_low, a, b)  # mínimo local\n        if Pmin > Pmax; Pmin, Pmax = Pmax, Pmin; end\n        @printf(\"  T=0.90 Tc: faixa de 3 soluções para P ∈ (%.5f, %.5f) atm\\n\", Pmin, Pmax)\n\n        # Escolhe P no meio da faixa e resolve as 3 raízes\n        Pmid = 0.5*(Pmin + Pmax)\n        Vroots = solve_V_for_P(Pmid, T_low, a, b)\n        if length(Vroots) == 3\n            @printf(\"  Exemplo P=%.5f atm: V1=%.6g < V2=%.6g < V3=%.6g (L/mol)\\n\",\n                    Pmid, Vroots[0+1], Vroots[1+1], Vroots[2+1])\n        else\n            @printf(\"  Aviso: não foram encontradas 3 raízes para P=%.5f (encontradas=%d)\\n\", Pmid, length(Vroots))\n        end\n\n        # Um P fora da faixa => solução única\n        P_out = Pmax + 0.1  # um pouco acima de Pmax\n        Vuniq = solve_V_for_P(P_out, T_low, a, b)\n        @printf(\"  Exemplo P=%.5f atm (fora da faixa): solução única V ≈ %.6g L/mol\\n\", P_out, Vuniq[1])\n    else\n        @printf(\"  (Não foram detectados dois extremos para T=0.90 Tc — verifique parâmetros)\\n\")\n    end\nend\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "\n## Exercício 4–4 — Garrafa térmica (estado estacionário)\n\nSistema a resolver:\n\\[\n\\begin{aligned}\nq_1 &= 10^{-9}\\,\\big[(T_0\\!+\\!273)^4 - (T_1\\!+\\!273)^4\\big],\\\\[4pt]\nq_2 &= 4\\,(T_1 - T_2),\\\\[4pt]\nq_3 &= 1.3\\,(T_2 - T_3)^{4/3},\n\\end{aligned}\n\\qquad \\text{com}\\quad q_1=q_2=q_3, \\; T_0=450^\\circ C,\\; T_3=25^\\circ C.\n\\]\n\nUsaremos **Newton-Raphson para sistemas** em \\((T_1,T_2)\\) impondo \\(q_1-q_2=0\\) e \\(q_2-q_3=0\\).\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "execution_count": null,
      "outputs": [],
      "source": [
        "\nT0 = 450.0\nT3 = 25.0\n\nq1(T1) = 1e-9 * ( (T0+273)^4 - (T1+273)^4 )\nq2(T1,T2) = 4*(T1 - T2)\nq3(T2) = 1.3 * (T2 - T3)^(4/3)\n\nF(x) = begin\n    T1, T2 = x\n    [\n        q1(T1) - q2(T1,T2);\n        q2(T1,T2) - q3(T2)\n    ]\nend\n\n# Jacobiano 2x2\nfunction J(x)\n    T1, T2 = x\n    # Derivadas:\n    # d/dT1 q1 = -1e-9 * 4*(T1+273)^3\n    dq1_dT1 = -4e-9 * (T1+273)^3\n    # d/dT1 q2 = 4 ; d/dT2 q2 = -4\n    dq2_dT1 = 4.0\n    dq2_dT2 = -4.0\n    # d/dT2 q3 = 1.3*(4/3)*(T2 - T3)^(1/3)\n    dq3_dT2 = 1.3*(4/3)*(T2 - T3)^(1/3)\n\n    # F1 = q1 - q2 => [dq1_dT1 - dq2_dT1,    -dq2_dT2]\n    # F2 = q2 - q3 => [dq2_dT1,              dq2_dT2 - dq3_dT2]\n    return [\n        dq1_dT1 - dq2_dT1   -dq2_dT2;\n        dq2_dT1             dq2_dT2 - dq3_dT2\n    ]\nend\n\nx0 = [200.0, 80.0]  # chute físico: T1 entre T0 e T2; T2 acima de T3\nsol, it = newton_system(F, J, x0; tol=1e-12, maxiter=100, log=false)\nT1, T2 = sol\n@printf(\"Solução: T1 ≈ %.6f °C,   T2 ≈ %.6f °C   (it=%d)\\n\", T1, T2, it)\n@printf(\"q1 = %.6e,  q2 = %.6e,  q3 = %.6e (W/m² em unidades coerentes do enunciado)\\n\",\n        q1(T1), q2(T1,T2), q3(T2))\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "\n### Observações finais\n\n- Resultados são sensíveis aos chutes iniciais nas abordagens abertas (Newton, Secante).  \n- O método de ponto fixo exige **contração** (\\(|g'(x)|<1\\)) na vizinhança da solução.  \n- Em Van der Waals, a presença de **três soluções** para \\(T<T_c\\) é delimitada pela faixa entre os extremos locais de \\(P(V)\\).\n\n> Execute cada seção e, se desejar, ajuste tolerâncias (`tol`) e limites de iterações (`maxiter`).\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Julia 1.10",
      "language": "julia",
      "name": "julia-1.10"
    },
    "language_info": {
      "file_extension": ".jl",
      "mimetype": "application/julia",
      "name": "julia",
      "pygments_lexer": "julia",
      "version": "1.10.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}