function [L, U, Q] = stewart_ik(pose,P)
%STEWART_IK Cinematica inversa de una plataforma Stewart 6-SPS.
% pose = [x;y;z;roll;pitch;yaw], angulos en rad.
% P.A: anclajes en plataforma movil; P.B: anclajes en base.
% L: longitudes de las seis patas.
% U: vectores unitarios de cada pata (base -> plataforma).
% Q: posiciones globales de anclajes de la plataforma.

pose = pose(:);
assert(numel(pose)==6,'pose debe ser 6x1.');
R = rpy2R(pose(4:6));
t = pose(1:3);
Q = (R*P.A.' + t).';
D = Q - P.B;
L = sqrt(sum(D.^2,2));
U = D ./ L;
end

function R = rpy2R(rpy)
r = rpy(1); p = rpy(2); y = rpy(3);
cr=cos(r); sr=sin(r); cp=cos(p); sp=sin(p); cy=cos(y); sy=sin(y);
Rx=[1 0 0;0 cr -sr;0 sr cr];
Ry=[cp 0 sp;0 1 0;-sp 0 cp];
Rz=[cy -sy 0;sy cy 0;0 0 1];
R = Rz*Ry*Rx;
end
