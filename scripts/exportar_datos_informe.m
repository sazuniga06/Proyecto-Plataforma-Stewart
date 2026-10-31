% exportar_datos_informe.m
% Exporta a informe/datos/ los resultados del motor de Octave que usan las figuras y tablas del
% informe. Ejecutar desde la raiz del proyecto:
%   octave --no-gui --quiet scripts/exportar_datos_informe.m
root = fileparts(fileparts(mfilename('fullpath')));
eng = fullfile(root, 'octave_engine');
addpath(eng); addpath(fullfile(eng, 'config')); addpath(fullfile(eng, 'kinematics'));
addpath(fullfile(eng, 'dynamics')); addpath(fullfile(eng, 'controllers')); addpath(fullfile(eng, 'solvers'));
out = fullfile(root, 'informe', 'datos');

P = stewart_params();
sys = linealizacion_analitica('mechanical', P);

% --- Cinematica: Jacobiano nominal y mapa de condicion
J0 = stewart_jacobian(P.pose0, P);
sv = svd(J0);
xs = linspace(-0.045, 0.045, 61); zs = linspace(P.Lmin, P.Lmax, 3);
cond_xy = zeros(numel(xs)); cond_rp = zeros(numel(xs));
angs = linspace(deg2rad(-18), deg2rad(18), numel(xs));
for i = 1:numel(xs)
    for j = 1:numel(xs)
        cond_xy(j, i) = cond(stewart_jacobian([xs(i); xs(j); P.h0; 0; 0; 0], P));
        cond_rp(j, i) = cond(stewart_jacobian([0; 0; P.h0; angs(i); angs(j); 0], P));
    end
end
[M0, ~, g0] = stewart_dynamic_matrices(P.pose0, zeros(6,1), P);
kin = struct('B', P.B, 'A', P.A, 'L0', P.L0, 'Lmin', P.Lmin, 'Lmax', P.Lmax, 'J0', J0, 'sv', sv, ...
             'condJ0', max(sv)/min(sv), 'detJ0', det(J0), 'manip', sqrt(det(J0*J0.')), ...
             'xs', xs, 'angs', angs, 'cond_xy', cond_xy, 'cond_rp', cond_rp, 'M0', M0, 'g0', g0);
save('-v7', fullfile(out, 'cinematica.mat'), '-struct', 'kin');

% --- Modelo linealizado
lin = struct('A', sys.A, 'B', sys.B, 'Ad', sys.Ad, 'Bd', sys.Bd, 'poles_cont', sys.poles_cont, ...
             'poles_disc', sys.poles_disc, 'u0', sys.u0, 'Fleg0', sys.Fleg0, ...
             'rank_Mc', sys.rank_Mc, 'rank_Mo', sys.rank_Mo, 'Ts', sys.Ts, 'sv_Mc', svd(sys.Mc), 'sv_Mo', svd(sys.Mo));
save('-v7', fullfile(out, 'lineal.mat'), '-struct', 'lin');

% --- Realimentacion de estados: diseno (factores 1) y optimo (results/ganancias_optimas.json)
opt = jsondecode(fileread(fullfile(root, 'results', 'ganancias_optimas.json')));
c1 = controlador_estado_observador(P, sys, 'integral_pi', struct());
c2 = controlador_estado_observador(P, sys, 'integral_pi', ...
     struct('pole_speed_factor', opt.estado.pole_scale, 'obs_speed_factor', opt.estado.obs_scale));
ss = struct('poles_d', c1.poles_cl, 'obs_d', c1.poles_obs, 'poles_o', c2.poles_cl, 'obs_o', c2.poles_obs, ...
            'K_d', c1.K, 'Ki_d', c1.Ki, 'Nr_d', c1.Nr, 'K_o', c2.K, 'Ki_o', c2.Ki, 'Nr_o', c2.Nr, 'Ld_o', c2.Ld, ...
            'err_Nr_cof', c2.err_Nr_cof, 'sep_error', c2.sep_error, 'sep_error_d', c1.sep_error, ...
            'pole_scale', opt.estado.pole_scale, 'obs_scale', opt.estado.obs_scale);
save('-v7', fullfile(out, 'estados.mat'), '-struct', 'ss');

fprintf('Datos exportados a %s\n', out);
