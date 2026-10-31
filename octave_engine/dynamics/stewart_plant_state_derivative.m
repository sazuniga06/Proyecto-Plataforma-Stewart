function [xdot, info] = stewart_plant_state_derivative(x, u, tau_d, P)
%STEWART_PLANT_STATE_DERIVATIVE Derivada de estado continua xdot = f(x, u) de la plataforma Stewart 6-SPS.
%
% Modelo de Lopes (2009) con la fuerza axial de cada pata como entrada:
%   x = [q; qdot] (12 estados),  u = F (6x1, N, saturada a +-P.act.Fmax)
%   M(q) qddot + C(q, qdot) qdot + g(q) = (J Tq)' F + tau_d
%
% Entradas:
%   x     - Vector de estado (12x1)
%   u     - Fuerzas axiales de las patas (6x1) [N]
%   tau_d - Perturbacion externa generalizada aplicada a la pose (6x1, opcional)
%   P     - Parametros del sistema (stewart_params)
%
% Salidas:
%   xdot - Derivada temporal del estado (12x1)
%   info - Diagnosticos (M, C, g, J, Tq, Ldot, F_leg, qddot)

if nargin < 3 || isempty(tau_d), tau_d = zeros(6,1); end
if nargin < 4 || isempty(P), P = stewart_params(); end

q = x(1:6); qdot = x(7:12);
[M, C, g, ~, J, Tq] = stewart_dynamic_matrices(q, qdot, P);
F_leg = max(min(u(:), P.act.Fmax), -P.act.Fmax);
qddot = M \ ((J * Tq).' * F_leg + tau_d(:) - C * qdot - g);
xdot = [qdot; qddot];

info.M = M;
info.C = C;
info.g = g;
info.J = J;
info.Tq = Tq;
info.qddot = qddot;
info.Ldot = J * Tq * qdot;
info.F_leg = F_leg;
end
