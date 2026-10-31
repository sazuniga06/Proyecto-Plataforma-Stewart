function [M, C, g, d, J, Tq, info] = stewart_dynamic_matrices(pose, qdot, P)
%STEWART_DYNAMIC_MATRICES Dinamica de la plataforma Stewart 6-SPS por el metodo del momento generalizado
% (A. M. Lopes, Commun. Nonlinear Sci. Numer. Simul. 14 (2009) 3389-3401). La entrada es la fuerza axial
% de cada pata:
%
%   M(q)*qddot + C(q,qdot)*qdot + g(q) = (J*Tq)' * F_leg + tau_ext
%
% Cuerpos: plataforma movil {P}; por pata i, cilindro {C_i} (unido a B_i, centro de masa a bC de B_i) y
% piston {S_i} (unido a P_i, centro de masa a bS de P_i). Cada cuerpo k tiene una velocidad
% u_k = J_k * nu, con nu = [v; omega] = Tq * qdot el twist de la plataforma, y su fuerza cinetica
% proyectada sobre {P} es J_k' * d/dt(I_k J_k Tq qdot) (ecs. 24, 44, 56, 66 y 74 de Lopes). Asi:
%   M = Tq' * sum_k J_k' I_k J_k * Tq,        C*qdot = Tq' * sum_k J_k' * d/dt(I_k J_k Tq) * qdot
%   g = dP/dq = Tq' * sum_k (J_k^lin)' * m_k * g * e_z        (energia potencial, ecs. 77-83)
%
% Jacobianos (base {B}, ecs. 34-40 y 60-62), con l_i = p + R s_i - b_i, p_i = R s_i, n_i = l_i/|l_i|:
%   J_Di = [lb~ , lb~ p_i~']           velocidad angular de la pata (lb = l_i / (l_i'l_i))
%   J_Bi = bC * n_i~' * J_Di           velocidad del centro de masa del cilindro
%   J_Gi = [I - bS n_i~' lb~] [I , p_i~']   velocidad del centro de masa del piston
%   J_i  = n_i' [I , p_i~']            velocidad de elongacion de la pata (fila i del Jacobiano)
%
% Salidas: M, C, g, d (= 0: el modelo de Lopes no tiene disipacion; se conserva por compatibilidad),
% J (Jacobiano geometrico, Ldot = J*nu), Tq (nu = Tq*qdot) e info (aportes por cuerpo).

if nargin < 3, P = stewart_params(); end
pose = pose(:); qdot = qdot(:);

[Ik, Jk, Tq, J] = cuerpos(pose, P);
Mnu = zeros(6);
for k = 1:numel(Jk)
    Mnu = Mnu + Jk{k}.' * Ik{k} * Jk{k};
end
M = Tq.' * Mnu * Tq;
M = 0.5 * (M + M.');

% Termino de Coriolis y centripeto: derivada temporal de I_k J_k Tq a lo largo de qdot (diferencia central)
h = 1e-6;
if any(qdot)
    [Ip, Jp, Tp] = cuerpos(pose + h*qdot, P);
    [Im, Jm, Tm] = cuerpos(pose - h*qdot, P);
    Vnu = zeros(6);
    for k = 1:numel(Jk)
        dIJT = (Ip{k} * Jp{k} * Tp - Im{k} * Jm{k} * Tm) / (2*h);
        Vnu = Vnu + Jk{k}.' * dIJT;
    end
    C = Tq.' * Vnu;
else
    C = zeros(6);
end

% Gravedad: gradiente de la energia potencial (cada cuerpo aporta J_lin' * m * g * e_z)
gz = [0; 0; P.gravity];
g_nu = [P.massTop * gz; zeros(3,1)];
[~, ~, Q] = stewart_ik(pose, P);
t = pose(1:3);
for i = 1:6
    [~, JB, JG] = jacobianos_pata(Q(i,:).' - t, Q(i,:).' - P.B(i,:).', P);
    g_nu = g_nu + JB.' * (P.leg.mC * gz) + JG.' * (P.leg.mS * gz);
end
g = Tq.' * g_nu;

d = zeros(6, 1);

if nargout > 6
    info.Mnu = Mnu;
    info.M_plataforma = Tq.' * (Jk{1}.' * Ik{1} * Jk{1}) * Tq;
    info.M_patas = M - info.M_plataforma;
    info.g_nu = g_nu;
end
end

% -----------------------------------------------------------------------------------------------------
function [Ik, Jk, Tq, J] = cuerpos(pose, P)
% Inercias I_k (en {B}) y Jacobianos J_k (u_k = J_k * nu) de todos los cuerpos en la pose dada.
R = rpy2R(pose(4:6));
p = pose(5); y = pose(6);
E = [cos(y)*cos(p), -sin(y), 0; sin(y)*cos(p), cos(y), 0; -sin(p), 0, 1];
Tq = blkdiag(eye(3), E);

[~, ~, Q] = stewart_ik(pose, P);
t = pose(1:3);
J = zeros(6);

Ik = {blkdiag(P.massTop * eye(3), R * P.platformInertiaBody * R.')};   % plataforma
Jk = {eye(6)};
for i = 1:6
    pi_ = Q(i,:).' - t;                      % R s_i
    li = Q(i,:).' - P.B(i,:).';              % vector de la pata
    [JD, JB, JG, ni, Ji] = jacobianos_pata(pi_, li, P);
    J(i,:) = Ji;
    Nn = ni * ni.';
    IC = P.leg.IC(1) * Nn + P.leg.IC(2) * (eye(3) - Nn);   % cilindro axisimetrico en {B}
    IS = P.leg.IS(1) * Nn + P.leg.IS(2) * (eye(3) - Nn);
    Ik(end+1:end+4) = {P.leg.mC * eye(3), IC, P.leg.mS * eye(3), IS};
    Jk(end+1:end+4) = {JB, JD, JG, JD};
end
end

function [JD, JB, JG, ni, Ji] = jacobianos_pata(pi_, li, P)
ni = li / norm(li);
lb = li / (li.' * li);
Sp = skew(pi_);
A = [eye(3), Sp.'];                          % velocidad de P_i: v + omega x p_i = A * nu
JD = skew(lb) * A;                           % ec. 36
JB = P.leg.bC * skew(ni).' * JD;             % ec. 40
JG = (eye(3) - P.leg.bS * skew(ni).' * skew(lb)) * A;   % ec. 62
Ji = ni.' * A;                               % fila del Jacobiano: [n_i' , (p_i x n_i)']
end

function S = skew(v)
S = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
end

function R = rpy2R(rpy)
cr = cos(rpy(1)); sr = sin(rpy(1)); cp = cos(rpy(2)); sp = sin(rpy(2)); cy = cos(rpy(3)); sy = sin(rpy(3));
R = [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr;
     sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr;
     -sp,   cp*sr,            cp*cr];
end
