function J = stewart_jacobian(pose,P)
%STEWART_JACOBIAN Jacobiano geometrico 6x6: Ldot = J*[v;omega].
% omega se interpreta en el marco base, coherente con la expresion
% instantanea r x u.

[~,U,Q] = stewart_ik(pose,P);
t = pose(1:3).';
J = zeros(6,6);
for i=1:6
    r = Q(i,:) - t;
    u = U(i,:);
    J(i,1:3) = u;
    J(i,4:6) = cross(r,u);
end
end
