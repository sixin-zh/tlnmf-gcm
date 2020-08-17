function [SDR,SIR,SAR,perm] = bss_eval2(se,s)
    addpath bss_eval21
    % Inputs:
    % se: nsrc x nsampl matrix containing estimated sources
    % s: nsrc x nsampl matrix containing true sources
    %
    % Outputs:
    % SDR: nsrc x 1 vector of Signal to Distortion Ratios
    % SIR: nsrc x 1 vector of Source to Interference Ratios
    % SAR: nsrc x 1 vector of Sources to Artifacts Ratios
    % perm: nsrc x 1 vector containing the best ordering of estimated sources
    % in the mean SIR sense (estimated source number perm(j) corresponds to
    % true source number j)
    
    %%% Errors %%%
    if nargin<2, error('Not enough input arguments.'); end
    [nsrc,nsampl]=size(se);
    [nsrc2,nsampl2]=size(s);
    if nsrc2~=nsrc, error('The number of estimated sources and reference sources must be equal.'); end
    if nsampl2~=nsampl, error('The estimated sources and reference sources must have the same duration.'); end
    
    SDR=zeros(nsrc,nsrc);
    SIR=zeros(nsrc,nsrc);
    SAR=zeros(nsrc,nsrc);
    for jest=1:nsrc,
        for jtrue=1:nsrc,
            [s_target, e_interf, e_artif] = bss_decomp_gain(se(jest,:), jtrue, s);
            [SDR(jest,jtrue), SIR(jest,jtrue), SAR(jest,jtrue)] = bss_crit(s_target, e_interf, e_artif);
            %[s_true,e_spat,e_interf,e_artif]=bss_decomp_mtifilt(se(jest,:),s,jtrue,512);
            %[SDR(jest,jtrue),SIR(jest,jtrue),SAR(jest,jtrue)]=bss_source_crit(s_true,e_spat,e_interf,e_artif);
        end
    end
    
    % Selection of the best ordering
    perm=perms(1:nsrc);
    nperm=size(perm,1); % commenter
    meanSIR=zeros(nperm,1);%
    for p=1:nperm,%
        meanSIR(p)=mean(SIR((0:nsrc-1)*nsrc+perm(p,:)));%
    end%
    [meanSIR,popt]=max(meanSIR);%
    perm=perm(popt,:).';%
    SDR=SDR((0:nsrc-1).'*nsrc+perm);%
    SIR=SIR((0:nsrc-1).'*nsrc+perm);%
    SAR=SAR((0:nsrc-1).'*nsrc+perm);%

end