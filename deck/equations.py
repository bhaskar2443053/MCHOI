"""LaTeX source of every equation in the pipeline deck (single source for OMML and PNG metrics)."""

EQUATIONS = {
    # Notation
    'n1': r'$\mathbf{p}_a=T_{a\leftarrow b}\,\mathbf{p}_b=R_{a\leftarrow b}\,\mathbf{p}_b+\mathbf{t}_{a\leftarrow b},\qquad T_{a\leftarrow c}=T_{a\leftarrow b}\,T_{b\leftarrow c}$',
    'n2': r'$\pi(K,D,\mathbf{X})=\left(f_x\,\tilde{x}+c_x,\;\;f_y\,\tilde{y}+c_y\right),\qquad (\tilde{x},\tilde{y})=d_D\!\left(X/Z,\;Y/Z\right)$',
    'n3': r'$d_D(x,y)=(1+k_1r^2+k_2r^4)\,(x,y)+\left(2p_1xy+p_2(r^2+2x^2),\;p_1(r^2+2y^2)+2p_2xy\right)$',
    'n4': r'$\mathrm{Exp}(\omega)=I+\frac{\sin\phi}{\phi}[\omega]_{\times}+\frac{1-\cos\phi}{\phi^2}[\omega]_{\times}^{2},\;\;\phi=\Vert\omega\Vert,\qquad \mathrm{Log}=\mathrm{Exp}^{-1}$',
    # Stage 1: calibration
    'c1': r'$\hat{\mathbf{u}}_{cik}=\pi\left(K_c,\,D_c,\;T_{c\leftarrow z}\,T_{z\leftarrow b_i}\,\mathbf{X}^{b}_{k}\right)$',
    'c2': r'$\min_{\{K_c,\,D_c,\,T_{c\leftarrow z}\},\;\{T_{z\leftarrow b_i}\}}\;\;\sum_{c,i,k}\;\Vert\mathbf{u}_{cik}-\hat{\mathbf{u}}_{cik}\Vert^{2}$',
    'c3': r'$\mathrm{RMS}_c=\sqrt{\frac{1}{N_c}\sum_{i,k}\Vert\mathbf{u}_{cik}-\hat{\mathbf{u}}_{cik}\Vert^{2}}$',
    # Stage 2: synchronization and export
    'y1': r'$\hat{\Delta}_c=\mathrm{arg\,min}_{\Delta}\;\sum_{k}\,d_{\mathrm{epi}}\left(\mathbf{u}^{c}_{k}(t_k+\Delta),\;\mathbf{u}^{z}_{k}(t_k)\right)^{2}$',
    'y2': r'$j^{*}(n)=\mathrm{arg\,min}_{j}\;\left|t^{c}_{j}-\hat{\Delta}_c-t^{z}_{n}\right|,\qquad \left|t^{c}_{j^{*}}-\hat{\Delta}_c-t^{z}_{n}\right|\leq 20\,\mathrm{ms}$',
    'y3': r'$\mathbf{X}_c(\mathbf{u})=s_c\,D_c(\mathbf{u})\;K_c^{-1}\,\left[u,\;v,\;1\right]^{\top}$',
    # Stage 3: detection and tracking
    'd1': r'$(R_m,\mathbf{t}_m)=\mathrm{arg\,min}\;\sum_{c}\sum_{j=1}^{4}\Vert\mathbf{u}_{cj}-\pi_c\left(T_{c\leftarrow z}(R_m\,\mathbf{c}_j+\mathbf{t}_m)\right)\Vert^{2}$',
    'd2': r'$\hat{x}_t=\alpha(f_c)\,x_t+\left(1-\alpha(f_c)\right)\hat{x}_{t-1},\qquad \alpha(f)=\frac{2\pi f\,\Delta t}{1+2\pi f\,\Delta t}$',
    'd3': r'$f_c=f_{\min}+\beta\,|\hat{v}_t|,\qquad \hat{v}_t=\alpha(f_d)\,\frac{x_t-x_{t-1}}{\Delta t}+\left(1-\alpha(f_d)\right)\hat{v}_{t-1}$',
    'd4': r'$\left(\theta_c,\;\mathbf{s}_c,\;\{\mathbf{u}_{cj},\kappa_{cj}\}_{j=1}^{21}\right)=\mathrm{WiLoR}\left(\mathrm{crop}(I_c,\;1.1\,\hat{\mathbf{b}}_c)\right)$',
    # Stage 4a: MANO fit
    'm1': r'$(\mathcal{V},\,\mathcal{J})=\mathcal{M}(\theta,\mathbf{s}),\qquad \theta=(R_0,R_1,\ldots,R_{15}),\;\;\mathbf{s}\in\mathbb{R}^{10}$',
    'm2': r'$\mathbf{J}_{tj}=\mathcal{J}_{j}(\theta_t,\mathbf{s})-\mathcal{J}_{0}(\theta_t,\mathbf{s})+\tau_t$',
    'm3': r'$\mathcal{L}=\mathcal{L}_{2D}+0.35\,\mathcal{L}_{3D}+0.15\,\mathcal{L}_{\mathrm{prior}}+4\,\mathcal{L}_{\mathrm{world}}+3\,\mathcal{L}_{\mathrm{finger}}+0.25\,\mathcal{L}_{\mathrm{rot}}$',
    'm4': r'$\mathcal{L}_{2D}=\sum_{t,c,j}w_{tcj}\,\rho_{15\,\mathrm{px}}\left(\pi_c(\mathbf{J}_{tj})-\mathbf{u}_{tcj}\right),\qquad \mathcal{L}_{3D}=\sum_{t,j}r_{tj}\,\rho_{2\,\mathrm{cm}}\left(\mathbf{J}_{tj}-\hat{\mathbf{X}}_{tj}\right)$',
    'm5': r'$\mathcal{L}_{\mathrm{world}}=\sum_{t}\rho_{6\,\mathrm{mm}}\left(\mathbf{J}_t-\left[(1-\eta_t)\mathbf{J}_{t-1}+\eta_t\mathbf{J}_{t+1}\right]\right),\qquad \eta_t=\frac{t-t_{-}}{t_{+}-t_{-}}$',
    # Stage 4b: rotation filter and arm
    'f1': r'$\omega_t=\frac{1}{\Delta t}\,\mathrm{Log}\left(R_t\,R_{t-1}^{\top}\right),\qquad f_c=f_{\min}+\beta\,\Vert\hat{\omega}_t\Vert$',
    'f2': r'$\hat{R}_t=\mathrm{Exp}\left(\alpha(f_c)\;\mathrm{Log}\left(R_t\,\hat{R}_{t-1}^{\top}\right)\right)\,\hat{R}_{t-1}$',
    'f3': r'$\mathbf{e}(\lambda)=\mathbf{o}_c+\lambda\,\mathbf{d}_c,\;\;\Vert\mathbf{e}(\lambda)-\mathbf{w}\Vert=L_f\;\Rightarrow\;\lambda=-b+\sqrt{b^2-\Vert\mathbf{o}_c-\mathbf{w}\Vert^2+L_f^2}$',
    'f4': r'$\hat{\mathbf{e}}=\mathbf{w}+L_f\,\frac{\mathrm{OE}(\mathbf{e})-\mathbf{w}}{\Vert\mathrm{OE}(\mathbf{e})-\mathbf{w}\Vert},\qquad \hat{\mathbf{h}}=\hat{\mathbf{e}}+L_u\,\frac{\mathrm{OE}(\mathbf{h})-\hat{\mathbf{e}}}{\Vert\mathrm{OE}(\mathbf{h})-\hat{\mathbf{e}}\Vert}$',
    'f5': r'$\mathbf{e}=\mathbf{h}+a\,\hat{\mathbf{n}}+\sqrt{L_u^2-a^2}\;\hat{\mathbf{p}},\qquad a=\frac{L_u^2-L_f^2+d^2}{2d},\;\;d=\Vert\mathbf{w}-\mathbf{h}\Vert$',
    # Stage 4c: object registration
    'o1': r'$T^{*}=\mathrm{arg\,max}_{T}\;\frac{1}{N}\sum_{n,c}\mathrm{IoU}\left(\mathcal{S}_{nc}(T),\,M_{nc}\right)-\lambda\,P_{\mathrm{rest}}(T),\qquad T=\left[R\,\mathrm{diag}(\sigma)\;|\;\mathbf{t}\right]$',
    'o2': r'$\mathcal{S}_{nc}(T)=\mathrm{hull}\left\{\pi_c\left(T_{c\leftarrow z}\,T_{z\leftarrow m}(n)\,T\,\mathbf{v}\right):\;\mathbf{v}\in\mathcal{V}\right\}$',
    'o3': r'$P_{\mathrm{rest}}=\frac{1}{2}\left[\min\left(\frac{\varphi}{30^{\circ}},1\right)+\min\left(\frac{\gamma}{3\,\mathrm{cm}},1\right)\right],\quad \gamma=\max(0,\,-1\,\mathrm{cm}-h_{\min},\,h_{\min}-4\,\mathrm{cm})$',
    'o4': r'$T_{z\leftarrow o}(t)=\mathrm{OE}\left(T_{z\leftarrow m}(t)\right)\;T^{*}$',
    # Stage 5: world frame and export
    'w1': r'$A=\left[\;\mathbf{f}\;;\;\mathbf{f}\times\mathbf{g}\;;\;\mathbf{g}\;\right],\qquad \mathbf{p}_w=100\,A\,(\mathbf{p}_z-\mathbf{o}),\qquad \mathbf{o}=-d_0\,\mathbf{n}_0$',
    'w2': r'$\mathbf{x}=\frac{\mathbf{p}-\mathbf{p}_{\mathrm{child}}}{\Vert\mathbf{p}-\mathbf{p}_{\mathrm{child}}\Vert},\;\;\mathbf{z}=\frac{\mathbf{k}-(\mathbf{k}\cdot\mathbf{x})\,\mathbf{x}}{\Vert\mathbf{k}-(\mathbf{k}\cdot\mathbf{x})\,\mathbf{x}\Vert},\;\;\mathbf{y}=\mathbf{z}\times\mathbf{x},\;\;q=\mathrm{quat}[\mathbf{x}\;\mathbf{y}\;\mathbf{z}]$',
    'w3': r'$\mathrm{ROT}=A\,R_{z\leftarrow o}\,F,\;\;F=\mathrm{diag}(1,-1,1),\qquad \mathbf{v}_w=\mathrm{LOC}+\mathrm{ROT}\;100\;F(\sigma\circ\mathbf{v})$',
    'w4': r'$\hat{\mathbf{u}}_c=\pi_c\left(T_{c\leftarrow z}\left(A^{\top}\mathbf{p}_w/100+\mathbf{o}\right)\right)$',
}
