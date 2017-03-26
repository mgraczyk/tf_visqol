function [moslqo, vnsim, debugInfo ] = visqol_simple(refSig, degSig, fs)
%
% ViSQOL
%
%
% Use NSIM to align and NSIM to calc similarity
% Use 16 critical bands for NB, 21 bands for WB
%
% (c) Andrew Hines, November 2012
%
%%

%critical bands (NB from ANSI SII spec)
bfs=[50 150 250 350 450 570 700 840 1000 1170 1370 1600 1850 2150 2500 2900 3400 4000 4800 6500 8000];
speechFlag=1;

NUM_BANDS=1:length(bfs);  % number of frequency bands to evaluate over
PATCH_SIZE=30;            % number of frames in a patch

windowsize=round((fs/8000)*256); %256 for 8k 512 for 16k sample wavs    
if rem(windowsize,2)~=0
    windowsize=windowsize-1;
end

window = hamming(windowsize, 'periodic');
   
% TODO(mgraczyk): Implement in Python.
%degSig=scaleSignalSPL(degSig,20*log10(sqrt(mean(refSig.^2)/20e-6))); % scale deg to ref signal intensity
    
%get spectrogram images for signals
img_rsig = getSigSpect(refSig,fs,bfs,window);
img_dsig = getSigSpect(degSig,fs,bfs,window);

%refloor relative to 0 for ref
% This bounds the range to be zero min for the ref signal.
reffloor=min(img_rsig(:));
degfloor=min(img_dsig(:));
lowfloor=reffloor;%min(reffloor,degfloor);

img_rsig=img_rsig-lowfloor;
img_dsig=img_dsig-lowfloor;
L=160; % Fix Dynamic range for speech to a constant for better mapping between datasets
[patches, refPatchIdxs]=createRefPatches(img_rsig, PATCH_SIZE);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%STEP FOUR: ALIGN DEGRADED PATCHES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% use alignment for audio where all patches should be almost aligned to
% begin with.
%[~, degPatchIdxs] = alignDegradedPatchesAudio(img_dsig, patches, NUM_BANDS,refPatchIdxs,L);

% TODO: Implement in python.
degPatchIdxs = refPatchIdxs;
       
replace_idx = abs(refPatchIdxs-degPatchIdxs) > 30;
degPatchIdxs(replace_idx) = refPatchIdxs(replace_idx);

vnsim = calcPatchSimilarity(patches, degPatchIdxs, img_dsig, NUM_BANDS, L, speechFlag);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%STEP SIX: PLOT FIGURE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%if plotFlag==1
    %plotVISQOL(img_rsig,img_dsig,patchcorr,PATCH_SIZE,refPatchIdxs,degPatchIdxs,patchNSIM,vnsim,bfs,t_sp_rsig,t_sp_dsig,moslqo,bandFlag);
 %end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%STEP SIX: Print Debug Info
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
debugInfo=[];

%debugInfo.bfs=bfs;
%debugInfo.bandFlag=bandFlag;
%debugInfo.refPatchIdxs=refPatchIdxs;
%debugInfo.degPatchIdxs=degPatchIdxs;
%debugInfo.t_sp=t_sp_rsig;
%debugInfo.hammingwindowsamples=windowsize;
%debugInfo.reffloor=reffloor;
%debugInfo.patchDeltas=patchDeltas;
%debugInfo.patchcorr=patchcorr;
%debugInfo.vnsim=vnsim;
%debugInfo.moslqo=moslqo;
%debugInfo.patchNSIM=patchNSIM;
%debugInfo.maxR=max(max(img_rsig));
%debugInfo.maxD=max(max(img_dsig));
%debugInfo.L=L;

moslqo = 0;
%[ moslqo ] = visqol2moslqo( vnsim );        

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%Internal functions
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [spec_bf] = getSigSpect(ssig, Fs_r, bfs, window)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INTERNAL FUNCTION: getSigSpect - this function generates the neurograms/spectrograms of the
% signal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    window_overlap = length(window)*0.5; 
    S = my_spectrogram(ssig, window, window_overlap, bfs, Fs_r);
    
    S=abs(S);                       % remove complex component
    S(S==0) = eps;                  % no -infs in power dB
    S = S/max(S(:));              % normalise Power
    spec_bf = 20*log10(S);          % power in dB
end


%%
function [patchcorr, degPatchIdxs]=alignDegradedPatchesAudio(img_dsig, patches, NUM_BANDS, refPatchIdxs,L)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INTERNAL FUNCTION: alignDegradedPatchesAudio - find the indices of the best patch matches in the
% degraded signal image.
%
% RETURNS
%   patchcorr correlation scores for each patch- used in plot of result
%   patchidx  this is the x-offset of the degraded patch best match to the
%             img_rsig patches
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %NSIM comparisons across reference signal with degraded signal

    %find the index of warp ratio 1:1
    NUM_FRAMES=size(patches{1,1},2);
    MAX_SLIDE_OFFSET=size(img_dsig,2)-size(patches{1,1},2);%use 1:1 ratio patch size
    NUM_PATCHES=length(patches(:,1));
    
    patchcorr=zeros(MAX_SLIDE_OFFSET,NUM_PATCHES); %patch correlation = min MSE with img_rsig patch
    degPatchIdxs=zeros(NUM_PATCHES,1);
    
    %step patch along time and find best match in img_dsig for each
    %img_rsig patch.
    
    startIdx=1; %initialise for fidx==1 

    for fidx=1:NUM_PATCHES % #number of patches
        
        
        if fidx>1
            startIdx=refPatchIdxs(fidx-1)+floor(size(patches{1,1},2)/2); %start at halfway through last patch 
        end
        if fidx<NUM_PATCHES
            endIdx=refPatchIdxs(fidx+1,1)-floor(size(patches{1,1},2)/2); 
        else
            endIdx=MAX_SLIDE_OFFSET;
        end
        
        slide_offset=startIdx;
        for idx=startIdx:endIdx
            img_patch=patches{fidx,1};
            patchcorr(idx,fidx)= nsim(img_patch(NUM_BANDS,1:NUM_FRAMES),img_dsig(NUM_BANDS,slide_offset:slide_offset+size(img_patch(:,1:NUM_FRAMES),2)-1),L,0);            
            slide_offset=slide_offset+1;
        end
        [~, degPatchIdxs(fidx)]=max(patchcorr(:,fidx));
    end
end

%%
function  [vnsim]=calcPatchSimilarity(patches, degPatchIdxs, img_dsig, NUM_BANDS, L,speechFlag)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INTERNAL FUNCTION: calculate the NSIM between all ref and degraded
% patches. For each reference patch, the NSIM is calculated between he
% degraded and the warped versions of the orginal signal. For each patch,
% the max NSIM is calculated
%
%
% RETURNS
%   patchcorr correlation scores for each patch- used in plot of result
%   patchidx  this is the x-offset of the degraded patch best match to the
%             img_rsig patches
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    NUM_PATCHES=length(patches(:,1));

    %get NSIM for patches
    for fidx=1:NUM_PATCHES % #number of patches
        sidx=1;
        for slide_offset=max(1,degPatchIdxs(fidx)):1:degPatchIdxs(fidx)
            img_patch=patches{fidx,1};
            if slide_offset+size(img_patch,2)-1>length(img_dsig(1,:)) %if slide goes past the end then don't try to compare. 
                mwxp(fidx,1,sidx)=0;%record as 0. the unslided version will be used anyway.
            else
                %start = slide_offset;
                %ee = min(slide_offset+size(img_patch,2)-1,length(img_dsig(1,:)));
                %disp([start,ee, ee-start])
                mwxp(fidx,1,sidx)= nsim(img_patch(NUM_BANDS,:),img_dsig(NUM_BANDS,slide_offset:min(slide_offset+size(img_patch,2)-1,length(img_dsig(1,:)))),L,speechFlag);
            end;
            sidx=sidx+1;
        end
    end
    
    for ppatchidx=1:NUM_PATCHES
        [~, opt_slide_idx]=max(max(mwxp(ppatchidx,:,:)));
        [~, opt_warp_idx]=max(mwxp(ppatchidx,:,opt_slide_idx));
        patchNSIM(ppatchidx)=mwxp(ppatchidx,opt_warp_idx,opt_slide_idx);
    end

    vnsim=mean(patchNSIM);
end

%%
function [patches, refPatchIdxs]=createRefPatches(img_rsig, PATCH_SIZE)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INTERNAL FUNCTION: create patches from reference signal image to test
% degraded against.
% For each patch frequency get patches based on max intensity at that f-band.
% For each warp test create patches reference patches warped by warp factor
% in "warps" input vector.
% percentages that can be used to test for warping in the degraded signal.
%
% Returns
%   patches     cell array of patches: rpws are patch #, cols are warped
%               versions
%
%   refPatchIdxs    array containing the x-offsets for the corresponding patch 
%               start indices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    refPatchIdxs=1:PATCH_SIZE:length(img_rsig)-PATCH_SIZE;%start 5 frames in to allow for warps
    %refPatchIdxs=PATCH_SIZE/2:PATCH_SIZE:length(img_rsig)-PATCH_SIZE;%start 5 frames in to allow for warps
    refPatchIdxs=refPatchIdxs';
    NUM_PATCHES=length(refPatchIdxs);
    patches=cell(NUM_PATCHES,1);    

    for fidx=1:NUM_PATCHES
        img_patch=img_rsig(:,refPatchIdxs(fidx):refPatchIdxs(fidx)+PATCH_SIZE-1); %patch window from echo signal to evaluate ref signal against
        [nrows, ncols]=size(img_patch);
        src_img_patch=img_patch;
        
        wcols=ncols;
        x1=(0:wcols-1)./(wcols-1).*(ncols-1);
        x1=x1+1;    
        img_patch= interp2((1:ncols), (1:nrows)', src_img_patch, x1,(1:nrows)','cubic');
        patches{fidx,1}=img_patch;
    end
    
end


%%
function [mNSIM, nmap] =nsim(neuro_r, neuro_d, L, speechFlag)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION nsim calc
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %set window size for NSIM comparison
    %window = fspecial('gaussian', [3 3], 0.5);
    window=    [0.0113    0.0838    0.0113
                0.0838    0.6193    0.0838
                0.0113    0.0838    0.0113];
            
    window = window/sum(sum(window));
    % dynamic range set to max of reference neuro
    %L=160;%
   % maxRD=max(max(max(neuro_r)), max(max(neuro_d)));
    % C1 and C2 constants
    K=[0.01 0.03];
    C1 = (K(1)*L)^2;
    C2 = ((K(2)*L)^2)/2;
    %Calc mean NSIM(r,d)
    neuro_r = double(neuro_r);
    neuro_d = double(neuro_d);
    mu_r   = filter2(window, neuro_r, 'valid');
    mu_d   = filter2(window, neuro_d, 'valid');
    mu_r_sq = mu_r.*mu_r;
    mu_d_sq = mu_d.*mu_d;
    mu_r_mu_d = mu_r.*mu_d;
    sigma_r_sq = filter2(window, neuro_r.*neuro_r, 'valid') - mu_r_sq;
    sigma_d_sq = filter2(window, neuro_d.*neuro_d, 'valid') - mu_d_sq;
    sigma_r_d = filter2(window, neuro_r.*neuro_d, 'valid') - mu_r_mu_d;
    sigma_r=sign(sigma_r_sq).*sqrt(abs(sigma_r_sq));
    sigma_d=sign(sigma_d_sq).*sqrt(abs(sigma_d_sq));
    L_r_d= (2*mu_r.*mu_d+C1) ./(mu_r_sq+mu_d_sq +C1);
    S_r_d= (sigma_r_d + C2)./(sigma_r.*sigma_d +C2);
    nmap=sign(L_r_d).*abs(L_r_d).*sign(S_r_d).*abs(S_r_d);
    
    %for speech just use a simple mean of the similarity map
    if speechFlag==true
        mNSIM =mean(nmap(:));
    else

    %for audio, handle LP filter issues of perfect simility for low bands
    %but missing high bands.
        patchFbandAvgSim=mean(nmap,2);
        if max(patchFbandAvgSim>=.999)==1
            nonPerfectBandMeans=[patchFbandAvgSim(patchFbandAvgSim<=.999)];
            if isempty(nonPerfectBandMeans)==1 % whole patch is a perfect match
                patchFbandAvgSim=ones(size(nmap,1),1);
            else
                patchFbandAvgSim=[nonPerfectBandMeans;1];
            end
        end

        mNSIM=mean(patchFbandAvgSim);
    end
    

end

%%
function [ moslqo ] = visqol2moslqo( nsim )

% NSIM to MOS-LQO mapping function - updated from ICASSP version after
% other code mods. Not used for ViSQOLAudio
py=[158.7423 -373.5843  295.5249  -75.2952];
moslqo = py(1).*nsim.^3 +py(2).*nsim.^2 + py(3)*nsim + py(4);

moslqo=min(5,moslqo);
moslqo=max(1,moslqo);
end

%%
function scaledSignal= scaleSignalSPL(InputSignal, SPL_required)

SPL_reference=20*log10(sqrt(mean(InputSignal.^2))/20e-6);
scale_factor = 10.^((SPL_required -SPL_reference)/20);
scaledSignal= InputSignal*scale_factor;
end

function [S] = my_spectrogram(x, window, window_overlap, bfs, Fs_r)
  % TODO(mgraczyk): Add tests for bit exactness with built in spectrogram.
  %S = spectrogram(x, window, window_overlap, bfs, Fs_r);

  num_blocks = ceil(size(x, 1) / window_overlap);
  num_result_blocks = floor(size(x, 1) / window_overlap) - 1;

  x_blocks = buffer(x, length(window), window_overlap, 'nodelay');
  x_blocks = window .* x_blocks;

  my_S = gga(x_blocks, bfs, Fs_r);
  S = my_S(:, 1:end-1);
end
