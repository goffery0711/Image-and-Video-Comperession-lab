clear;
path(path,'analysis')

%% Set parameter for qScale
k=1;
range = [0.07, 0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5];
for j = 1:numel(range)
    qScale = range(j);

    %%%%%%%%%%%%%% Still Image %%%%%%%%%%%%%%
    img = double(imread('data/sequences/foreman20_40_RGB/foreman0020.bmp'));
    % Use Lena_small
    [PSNR_frame_1, bitrate_frame_1, num_irr, recon_1st_frame] = Milestone_E41(img,0,701,qScale);
    
    % prepare for video implementation
    frame_1_rec_ycbcr = ictRGB2YCbCr(recon_1st_frame);
    frame_dec{1} = frame_1_rec_ycbcr;
    data_name = 'data/sequences/foreman20_40_RGB/foreman00';
    for i = 20 :40
        img_name = strcat(data_name, int2str(i),'.bmp');
        frame{i-19} = ictRGB2YCbCr(double(imread(img_name)));
    end
    motion_index_frame_1 = SSD(frame_dec{1}, frame{2});
    frame_2_rec = SSD_rec(frame_dec{1}, motion_index_frame_1);
    for i = 1:3
        frame_2_differ(:,:,i) = frame_2_rec(:,:,i) - frame{2}(:,:,i);
    end
    h_mi = histogram(motion_index_frame_1(:),1:82,'Normalization','probability');
    pmf_mi = h_mi.Values;
    [BinaryTree_mi, HuffCode_mi, BinCode_mi, Codelengths_mi] = buildHuffman(pmf_mi);
    Encoded_differ = Intra_encode_extra(frame_2_differ, qScale);
    h_differ = histogram(Encoded_differ(:),-600:1100,'Normalization','probability');
    pmf_differ = h_differ.Values;
    [BinaryTree_differ, HuffCode_differ, BinCode_differ, Codelengths_differ] = buildHuffman(pmf_differ);

    % Obtain bitrate and calculate PSNR
    Still_bitrate{j} = bitrate_frame_1;
    Still_PSNR{j} = PSNR_frame_1;
    bitrate_frame{1} = bitrate_frame_1;
    PSNR{1} = PSNR_frame_1;

    %%%%%%%%%%%%%% Video sequence %%%%%%%%%%%%%%
    for i = 2: length(frame)
        ref_im = frame_dec{i-1};
        im = frame{i};
        motion_index = SSD(ref_im, im);
        im_rec = SSD_rec(ref_im, motion_index);
        Differ = im - im_rec;
        bit_num_mi = codeLength(motion_index(:), Codelengths_mi);

        % E4Mulestone.m
        [psnr_frame, bitrate_img_only, bitnum_frame, frame_rec_differ] = Milestone_E41(Differ, Codelengths_differ, 601,qScale);
        
        % Obtain bitrate
        bits_number = bit_num_mi + bitnum_frame;
        bitrate_frame{i} = bits_number/(numel(frame{i})/3);
        frame_dec{i} = im_rec + frame_rec_differ;

        % Calculate PSNR
        recon_RGB = ictYCbCr2RGB(frame_dec{i});
        Ori_RGB = ictYCbCr2RGB(frame{i});
        MSE = calcMSE(Ori_RGB, recon_RGB);
        PSNR{i} = calcPSNR(MSE);
    end
    Final_bitrate{k} = mean([bitrate_frame{1:length(frame)}]);
    Final_PSNR{k} = mean([PSNR{1:length(frame)}]);
    k=k+1
end
%% plot RD curve
a = cell2mat(Final_bitrate);
b = cell2mat(Final_PSNR);
plot(a, b, 'bx-')
xlabel("bpp");
ylabel('PSNR [dB]');

hold on;
c = cell2mat(Still_bitrate);
d = cell2mat(Still_PSNR);
plot(c, d, 'rx-')
set(gca,'XTick', 0.0:0.5:6);

legend('Video Codec', 'Still image Codec')
title( 'R-D plot' ) ;

%% All Functions
function [psnr, bitrate, bitnum, recon] = Milestone_E41(Lena, Codelengths, bias, qScale)
lena_small = double(imread('data/images/lena_small.tif'));
flag = bias - 701;
scaleIdx = 1;
    %% Training Huffman using Lena_small
    if flag == 0     
        % use pmf of k_small to build and train huffman table
        IE_blk = [];
        for i = 1:8
           for j = 1:8
                block_s = lena_small((i-1)*8+1:8*i,(j-1)*8+1:8*j,:);
                IE_s = IntraEncode(block_s, qScale,0);
                IE_blk =[IE_blk;IE_s{1};IE_s{2};IE_s{3}];
            end
        end
        h = histogram(IE_blk(:),-700:2000,'Normalization','probability');
        pmf = h.Values;
        [BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman(pmf);
    end

    %% use trained table to encode k to get the bytestream
    num = 0;
    for i = 1:size(Lena,1)/8
        for j = 1:size(Lena,2)/8
            block_l = Lena((i-1)*8+1:8*i,(j-1)*8+1:8*j,:);
            IE_l = IntraEncode(block_l,qScale, flag);
            for n = 1:3
                bit_length = codeLength(IE_l{n}+bias, Codelengths);
                num = num + bit_length;
            end
            blk_rec = IntraDecode(IE_l, qScale);
            recon((i-1)*8+1:8*i,(j-1)*8+1:8*j,:) = blk_rec;
            
        end
    end
    bitnum = num;
    if flag == 0
        I_rec = ictYCbCr2RGB(recon);
    else
        I_rec = recon;
    end
    bitPerPixel(scaleIdx) = num/ (numel(Lena)/3);
    mse = calcMSE(Lena,I_rec);
    PSNR(scaleIdx) = calcPSNR(mse);
    %fprintf('QP: %.1f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', qScale, bitPerPixel(scaleIdx), PSNR(scaleIdx))
psnr = PSNR(1);
bitrate = bitPerPixel;
recon = I_rec;
end

function Encoded = Intra_encode_extra(image_differ, qscale)
% Input:  image_differ in YCbCr
% Output: Encoded image_differ into (Sequence after run-length coding)
    Encoded = [];
    for i = 1: size(image_differ,1)/8
        for j = 1: size(image_differ,2)/8
            blk = image_differ((i-1)*8+1:8*i,(j-1)*8+1:8*j,:);
            encoded_layer = IntraEncode(blk, qscale, 1);
            Encoded = [Encoded; encoded_layer{1}; encoded_layer{2}; encoded_layer{3}];
        end
    end
end

function dst = IntraEncode(image_block, qScale, flag)
%  Function Name : IntraEncode.m
%  Input         : image (Original RGB Image)
%                  qScale(quantization scale)
%  Output        : dst   (sequences after zero-run encoding, Nx3)
    % Convert to YCbCr
    if flag == 0
        img_ycbcr_block = ictRGB2YCbCr(image_block);
    else
        img_ycbcr_block = image_block;
    end
    % DCT
    %I_dct = blockproc(img_ycbcr, [8, 8], @(block_struct) process(block_struct.data, qScale));
    coeff = DCT8x8(img_ycbcr_block);
    quant = Quant8x8(coeff, qScale);
    zz = ZigZag8x8(quant);
    for i = 1:3
        zrc{i} = ZeroRunEnc_EoB(zz(:,i),999)';
    end
    dst = zrc;
    
end

function dst = IntraDecode(image , qScale)
%  Function Name : IntraDecode.m
%  Input         : image (zero-run encoded image, Nx3)
%                  img_size (original image size)
%                  qScale(quantization scale)
%  Output        : dst   (decoded image)
    for i = 1:3
        dec_zz(:,i) = ZeroRunDec_EoB(image{i}, 999);
    end
    quanted = DeZigZag8x8(dec_zz);
    coeff = DeQuant8x8(quanted,qScale);
    dst = IDCT8x8(coeff);
end

function zrc = process(block, qscale)
    coeff = DCT8x8(block);
    quant = Quant(coeff, qscale);
    zz = ZigZag8x8(quant);
    zrc = ZeroRunEnc_EoB(zz,99);
    zrc=[1 1 1];
    
    
    
end

function coeff = DCT8x8(block)
%  Input         : block    (Original Image block, 8x8x3)
%
%  Output        : coeff    (DCT coefficients after transformation, 8x8x3)
    [bh, bw, bd] = size(block);
    coeff = zeros([bh, bw, bd]);
    
    for i = 1:1:bd
        a = dct(block(:,:,i));
        coeff(:,:,i) = dct(a')';
    end
    
end

function quant = Quant8x8(dct_block, qScale)
%  Input         : dct_block (Original Coefficients, 8x8x3)
%                  qScale (Quantization Parameter, scalar)
%
%  Output        : quant (Quantized Coefficients, 8x8x3)

   L =  qScale*[16, 16, 16, 16, 17, 18, 21, 24;
                16, 16, 16, 16, 17, 19, 22, 25;
                16, 16, 17, 18, 20, 22, 25, 29;
                16, 16, 18, 21, 24, 27, 31, 36;
                17, 17, 20, 24, 30, 35, 41, 47;
                18, 19, 22, 27, 35, 44, 54, 65;
                21, 22, 25, 31, 41, 54, 70, 88;
                24, 25, 29, 36, 47, 65, 88, 115];
                           
    C =  qScale*[17, 18, 24, 47, 99, 99, 99, 99;
                 18, 21, 26, 66, 99, 99, 99, 99;
                 24, 26, 56, 99, 99, 99, 99, 99;
                 47, 66, 99, 99, 99, 99, 99, 99;
                 99, 99, 99, 99, 99, 99, 99, 99;
                 99, 99, 99, 99, 99, 99, 99, 99;
                 99, 99, 99, 99, 99, 99, 99, 99;
                 99, 99, 99, 99, 99, 99, 99, 99];
    
     quant = zeros(size(dct_block));
     quant(:, :, 1) = round(dct_block(:, :, 1) ./ L);
     quant(:, :, 2) = round(dct_block(:, :, 2) ./ C);
     quant(:, :, 3) = round(dct_block(:, :, 3) ./ C);
end    

function zz = ZigZag8x8(quant)
%  Input         : quant (Quantized Coefficients, 8x8x3)
%
%  Output        : zz (zig-zag scaned Coefficients, 64x3)
    zz = zeros(64, 3);
    zz_indices = [1     2     6     7     15   16   28   29;
                  3     5     8     14    17   27   30   43;
                  4     9     13    18    26   31   42   44;
                  10    12    19    25    32   41   45   54;
                  11    20    24    33    40   46   53   55;
                  21    23    34    39    47   52   56   61;
                  22    35    38    48    51   57   60   62;
                  36    37    49    50    58   59   63   64];
    for ch = 1:1:3
       q_ch = quant(:, :, ch);
       zz(zz_indices, ch) = q_ch(:);
    end
end

function zze = ZeroRunEnc_EoB(zz, EOB)
%  Input         : zz (Zig-zag scanned block, 1x64)
%                  EOB (End Of Block symbol, scalar)
%
%  Output        : zze (zero-run-level encoded block, 1xM)
    prev_is_zero = 0;    
    zero_counting = 0;
    zze = [];
    zz_len = length(zz);
    
    for i = 1:1:zz_len
        q = zz(i);
        if q ~= 0
            if prev_is_zero == 1
                zze = [zze, 0, zero_counting];
            end
            zze = [zze, q];
            prev_is_zero = 0;
        else
            if prev_is_zero == 1
                zero_counting = zero_counting + 1;
            else
                prev_is_zero = 1;
                zero_counting = 0;
            end
            if i == zz_len | mod(i,64) == 0
                zze = [zze, EOB];
                prev_is_zero = 0;    
                zero_counting = 0;
            end
        end
    end
end

function dst = ZeroRunDec_EoB(src, EoB)
%  Function Name : ZeroRunDec1.m zero run level decoder
%  Input         : src (zero run encoded sequence 1xM with EoB sign in the end)
%                  EoB (end of block sign)
%
%  Output        : dst (reconstructed single zig-zag scanned block 1x64)
    src_len = length(src);
    dst = [];
    dec_len = 0;
    zero_found = 0;
    
    for i = 1:1:src_len
       q = src(i);
       if q == EoB
           if mod(dec_len, 64) == 0
              dst(dec_len+1:dec_len+64) = 0;
              dec_len = dec_len + 64;
           else
               a = 64 - mod(dec_len, 64);
               dst(dec_len+1:dec_len+a) = 0;
               dec_len = dec_len + a;
           end
       elseif q == 0
           if zero_found 
               % only 1 zero symbol
               dst(dec_len+1) = 0;
               dec_len = dec_len + 1;
               zero_found = 0;
           else
               zero_found = 1;
           end
       else
           if zero_found
               % this symbol represents number of zeros
               dst(dec_len+1:dec_len+1+q) = 0;
               dec_len = dec_len + q + 1;
           else
               dst(dec_len+1) = q;
               dec_len = dec_len + 1;
               
           end
           
           zero_found = 0;
       end
    end
    %dst = dst(1,:);
end

function coeffs = DeZigZag8x8(zz)
%  Function Name : DeZigZag8x8.m
%  Input         : zz    (Coefficients in zig-zag order)
%
%  Output        : coeffs(DCT coefficients in original order)
    zz_indices = [1     2     6     7     15   16   28   29;
                  3     5     8     14    17   27   30   43;
                  4     9     13    18    26   31   42   44;
                  10    12    19    25    32   41   45   54;
                  11    20    24    33    40   46   53   55;
                  21    23    34    39    47   52   56   61;
                  22    35    38    48    51   57   60   62;
                  36    37    49    50    58   59   63   64];
    
    [~, chs] = size(zz);
    coeffs = zeros(8, 8, chs);
    for ch = 1:1:chs
        zz_ch = zz(:, ch);
        coeffs(:, :, ch) = reshape(zz_ch(zz_indices(:)), 8, 8);
    end
end

function dct_block = DeQuant8x8(quant_block, qScale)
%  Function Name : DeQuant8x8.m
%  Input         : quant_block  (Quantized Block, 8x8x3)
%                  qScale       (Quantization Parameter, scalar)
%
%  Output        : dct_block    (Dequantized DCT coefficients, 8x8x3)
    L =  qScale*[16, 16, 16, 16, 17, 18, 21, 24;
                 16, 16, 16, 16, 17, 19, 22, 25;
                 16, 16, 17, 18, 20, 22, 25, 29;
                 16, 16, 18, 21, 24, 27, 31, 36;
                 17, 17, 20, 24, 30, 35, 41, 47;
                 18, 19, 22, 27, 35, 44, 54, 65;
                 21, 22, 25, 31, 41, 54, 70, 88;
                 24, 25, 29, 36, 47, 65, 88, 115];

    
    C =  qScale*[17, 18, 24, 47, 99, 99, 99, 99;
                 18, 21, 26, 66, 99, 99, 99, 99;
                 24, 26, 56, 99, 99, 99, 99, 99;
                 47, 66, 99, 99, 99, 99, 99, 99;
                 99, 99, 99, 99, 99, 99, 99, 99;
                 99, 99, 99, 99, 99, 99, 99, 99;
                 99, 99, 99, 99, 99, 99, 99, 99;
                 99, 99, 99, 99, 99, 99, 99, 99];
    
     dct_block = zeros(size(quant_block));
     dct_block(:, :, 1) = (quant_block(:, :, 1) .* L);
     dct_block(:, :, 2) = (quant_block(:, :, 2) .* C);
     dct_block(:, :, 3) = (quant_block(:, :, 3) .* C);

end

function block = IDCT8x8(coeff)
%  Function Name : IDCT8x8.m
%  Input         : coeff (DCT Coefficients) 8*8*3
%  Output        : block (original image block) 8*8*3
    [bh, bw, bd] = size(coeff);    
    block = zeros([bh, bw, bd]);

    for i = 1:1:bd
        a = idct(coeff(:,:,i));
        block(:,:,i) = idct(a')';
    end
end



function CL = codeLength(data, Codelengths)
    CL = 0;
    for i = 1:1:length(data)
        index = floor(data(i));
        if index < 1
            index = 1;
        end
        CL = CL + Codelengths(index);
    end


end

function motion_vectors_indices = SSD(ref_image, image)
%  Input         : ref_image(Reference Image, size: height x width)
%                  image (Current Image, size: height x width)
%
%  Output        : motion_vectors_indices (Motion Vector Indices, 
%  size: (height/8) x (width/8) x 1 )
    
    % Pad the ref_image with 4x4 Boarder
    ref = padarray(ref_image,[4,4],0,'both');
    ref = ref(:,:,1);
    image = image(:,:,1);
    % Processing with blocks
    motion_vectors_indices = blockproc(image, [8, 8], @(block_struct) ...
        find_index(block_struct.data, block_struct.location, ref));   

end

% Function to transform MV to MV index
function index = mv2index(mv)
    % +-4 Search range
    row = mv(1) + 5;
    col = mv(2) + 5;
    index = (row - 1)*9 + col;
end

% Function to find MV_index under 8x8 block
function index = find_index(block, location, ref)
    % Search from -4 to +4
    % location of ref changed because of the boarders
    best_ssd = Inf;
    loc_ref = location + [4, 4];
    for i = -4 : 4
        for j = -4 : 4
            % Motion vector as follow
            mv = [i, j];
            % Calc SSD
            ref_block = ref(loc_ref(1) + i : loc_ref(1) + i + 7,loc_ref(2)+j: loc_ref(2)+j+7);
            ssd = sum(sum((block - ref_block).^2));
            % Compare with the best_ssd
            if ssd < best_ssd
                best_ssd = ssd;
                best_mv = mv;
            end
        end
    end
    index = mv2index(best_mv);
    %index = best_ssd;
end


function rec_image = SSD_rec(ref_image, motion_vectors)
%  Input         : ref_image(Reference Image, YCbCr image)
%                  motion_vectors
%
%  Output        : rec_image (Reconstructed current image, YCbCr image)

    rec_image = zeros(size(ref_image));
    ref_image = padarray(ref_image,[4 4],0,'both');
    for i = 1: size(motion_vectors,1)
        for j = 1: size(motion_vectors,2)
            mv = index2mv(motion_vectors(i,j));
            block_rec = ref_image((i-1)*8+5 + mv(1):(i-1)*8+1 + mv(1) + 11, ...
                (j-1)*8+5+mv(2):(j-1)*8 + 1 + mv(2) + 11,:);
            rec_image((i-1)*8+1:(i-1)*8+8, (j-1)*8+1:(j-1)*8+8,:) = block_rec;
        end
    end
end

function mv = index2mv(index)
    % Get the loc of the MV-Matrix
    col = mod(index,9);
    row = fix(index/9)+1;
    if col == 0
        col =9;
        row = row - 1;
    end
    mv(1) = row -5;
    mv(2) = col -5;
end


function PSNR = calcPSNR(MSE)
    PSNR = 10*log10((2.^8-1)^2/MSE);
end

function MSE = calcMSE(Image, recImage)
    [h, w, c] = size(Image);
    img1 = double(Image);
    img2 = double(recImage);
    MSE = sum(sum(sum( (img1 - img2).^2)))/h/w/c;
end

function yuv = ictRGB2YCbCr(rgb)
    
    [h, w, c] = size(rgb);
    if c ~= 3
        error("image must have 3 channels");
    end
    yuv = zeros(h, w, c);
    
    for r = 1:h
        for d = 1:w
            yuv(r, d, 1) = [0.299, 0.587, 0.114]*[rgb(r, d, 1);rgb(r, d, 2);rgb(r, d, 3)];
            yuv(r, d, 2) = [-0.169,-0.331, 0.5]*[rgb(r, d, 1);rgb(r, d, 2);rgb(r, d, 3)];
            yuv(r, d, 3) = [0.5,- 0.419, - 0.081]*[rgb(r, d, 1);rgb(r, d, 2);rgb(r, d, 3)];
        end
    end
    
end

function rgb = ictYCbCr2RGB(yuv)

    [h, w, c] = size(yuv);
    if c ~= 3
        error("image must have 3 channels");
    end
    rgb = zeros(h, w, c);
    
    for r = 1:h
        for d = 1:w
            rgb(r, d, 1) = [1, 0, 1.402]*[yuv(r, d, 1);yuv(r, d, 2);yuv(r, d, 3)];
            rgb(r, d, 2) = [1, -0.344, -0.714]*[yuv(r, d, 1);yuv(r, d, 2);yuv(r, d, 3)];
            rgb(r, d, 3) = [1, 1.772, 0]*[yuv(r, d, 1);yuv(r, d, 2);yuv(r, d, 3)];
            
        end
    end

end
