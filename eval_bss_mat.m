addpath bss_eval21
L = 10;
K = 2;
SIMs = 5;
SDRs = zeros(SIMs,K);
SIRs = zeros(SIMs,K);
SARs = zeros(SIMs,K);
for sim = 1:SIMs
    mat_srcs = sprintf('./datasets/nonstationary440_sim%d.mat',sim);   
    % eval DCT-NMF
    mat_recs_pre = sprintf('./results_dct2/nonstationary440_sim%d_win4_ws40ms_run1/isnmf_dctsci_batch_nonstationary440_sim%d_K2_eps5e-07',sim,sim);
    % eval TL-NMF
    % mat_recs_pre = sprintf('./results_pertl20/nonstationary440_sim%d_win4_ws40ms_run1/tlnmf2_sci_batch_nonstationary440_sim%d_K2_eps5e-07',sim,sim);
    
    load(mat_srcs,'ys1','ys2')
    assert(K==2)
    y1rs = zeros(size(ys1));
    y2rs = zeros(size(ys2));
    for l=1:L
        for k=1:K
            mat_recs = sprintf('%s_l%d_piece%d.mat',mat_recs_pre,l-1,k-1);
            load(mat_recs,'y_k')
            if k==1
                y1rs(l,:) = y_k;
            else
                y2rs(l,:) = y_k;
            end
        end
    end

    %% eval
    SDRc_sum = zeros(K,1);
    SIRc_sum = zeros(K,1);
    SARc_sum = zeros(K,1);
    for l=1:L
        [SDRc,SIRc,SARc,~] = bss_eval2([y1rs(l,:);y2rs(l,:)],[ys1(l,:);ys2(l,:)]); % Perfos of one sample l        
        SDRc_sum = SDRc_sum+SDRc;
        SIRc_sum = SIRc_sum+SIRc;
        SARc_sum = SARc_sum+SARc;               
    end
    SDRs(sim,:) = SDRc_sum / L;
    SIRs(sim,:) = SIRc_sum / L;
    SARs(sim,:) = SARc_sum / L;    
end

%%
fprintf('Average (%d samples) Performances BSS\n',L)
for k=1:K
	fprintf('$%d$ & %.2f (%.2f) &  %.2f (%.2f) & %.2f (%.2f) \\\\ \n', ...
            k, mean(SDRs(:,k)), std(SDRs(:,k)), mean(SIRs(:,k)), std(SIRs(:,k)), ...
            mean(SARs(:,k)), std(SARs(:,k)) );         
end
