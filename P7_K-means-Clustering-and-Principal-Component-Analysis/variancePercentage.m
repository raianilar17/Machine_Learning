function [per, S_per] = variancePercentage(X_norm, X_rec, S, K)
  
  m = size(X_norm,1);
  Avg_Sqr_Proj_err = ((1/m)*(sum(sum(((X_norm .- X_rec).^2),2))));
  Total_Var = (1/m)*(sum(sum((X_norm.^2),2)));
  val = (Avg_Sqr_Proj_err / Total_Var); 
  per = (1 - val)*100;
  
  s_k = sum(sum(S(:,1:K)));
  s_n = sum(sum(S));
  
  S_per = (s_k / s_n)* 100;
  
end;
