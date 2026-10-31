function [K, V, res_poles] = octave_place_mimo(A, B, desired_poles)
%OCTAVE_PLACE_MIMO Asignacion de polos multivariable (MIMO) por ecuaciones de Sylvester.
%
% Calcula la matriz de ganancia K tal que eig(A - B*K) == desired_poles.
% Utiliza el metodo de asignacion estructural de autovectores de Sylvester
% (Moore 1976, Kautsky-Nichols-Van Dooren 1985):
%   (A - p_j * I) * v_j = B * g_j,   K = real(G * V^(-1))
%
% Agrupa cada canal de entrada c = 1..m con sus polos correspondientes y
% asegura que los pares complejos conjugados compartan exactamente la misma
% direccion tangencial g_c, garantizando K estrictamente real y bien condicionada.
%
% Entradas:
%   A             - Matriz de estado (n x n)
%   B             - Matriz de entrada (n x m)
%   desired_poles - Vector de n polos deseados (reales o pares conjugados complejos)
%
% Salidas:
%   K         - Ganancia de realimentacion (m x n)
%   V         - Matriz modal de autovectores de lazo cerrado (n x n)
%   res_poles - Polos efectivos obtenidos en lazo cerrado eig(A - B*K)

n = size(A, 1);
m = size(B, 2);
desired_poles = desired_poles(:);

if numel(desired_poles) ~= n
    error('El vector desired_poles debe tener longitud igual a n=%d.', n);
end

% Direcciones de entrada desacopladas por canal si el bloque inferior es cuadrado e invertible
G_dirs = eye(m);
if n >= 2*m && m == 6
    B_lower = B(m+1:2*m, :);
    if rank(B_lower) == m
        B_inv = inv(B_lower);
        for c = 1:m
            G_dirs(:, c) = B_inv(:, c) / norm(B_inv(:, c));
        end
    end
end

V = zeros(n, n);
G = zeros(m, n);
I_n = eye(n);

% Asignacion ciclica por canales manteniendo pares conjugados en el mismo canal
c_idx = 0;
j = 1;
while j <= n
    c = mod(c_idx, m) + 1;
    c_idx = c_idx + 1;
    g_base = G_dirs(:, c);
    
    p_j = desired_poles(j);
    if abs(imag(p_j)) > 1e-9 && j < n && abs(p_j - conj(desired_poles(j+1))) < 1e-7
        % Par complejo conjugado en el mismo canal c
        g_j = g_base;
        v_j = (A - p_j * I_n) \ (B * g_j);
        V(:, j)   = v_j;
        G(:, j)   = g_j;
        V(:, j+1) = conj(v_j);
        G(:, j+1) = g_j;
        j = j + 2;
    else
        % Polo real: pequena perturbacion determinista por capa para evitar colinealidad si hay multiples polos reales en el mismo canal
        layer = floor((j - 1) / m);
        g_j = g_base;
        if layer > 0
            c_next = mod(c, m) + 1;
            g_j = g_base + 0.05 * layer * G_dirs(:, c_next);
            g_j = g_j / norm(g_j);
        end
        v_j = (A - p_j * I_n) \ (B * g_j);
        V(:, j) = v_j;
        G(:, j) = g_j;
        j = j + 1;
    end
end

K = real(G / V);
res_poles = eig(A - B * K);
end
