clear

load('../extracted_feature_iter50000/img_fea.mat')
img_fea = exp(x);

load('../extracted_feature_iter50000/txt_fea.mat')
txt_fea = exp(x);

load('../extracted_feature_iter50000/aud_fea.mat')
aud_fea = exp(x);

load('../extracted_feature_iter50000/vid_fea.mat')
vid_fea = exp(x);

load('../extracted_feature_iter50000/td_fea.mat')
td_fea = exp(x);


load('./listFile/test_img_txt_lab.mat');
img_lab = test_lab;
txt_lab = test_lab;
importdata('./listFile/test_aud.txt');
aud_lab = ans.data;
load('./listFile/test_vid_lab.mat');
vid_lab = test_lab;
importdata('./listFile/test_3d.txt');
td_lab = ans.data;

[a,b]=sort(img_lab);
img_fea=img_fea(b,:);
img_lab=img_lab(b);


[a,b]=sort(txt_lab);
txt_fea=txt_fea(b,:);
txt_lab=txt_lab(b);

[a,b]=sort(aud_lab);
aud_fea=aud_fea(b,:);
aud_lab=aud_lab(b);

[a,b]=sort(vid_lab);
vid_fea=vid_fea(b,:);
vid_lab=vid_lab(b);


[a,b]=sort(td_lab);
td_fea=td_fea(b,:);
td_lab=td_lab(b);

te_n_I = 8000;
te_n_T = 8000;
te_n_A = 2000;
te_n_V = 1000;
te_n_TD = 400;

D = pdist([img_fea; txt_fea; aud_fea; vid_fea; td_fea],'Cos');
W = -squareform(D);



load('../extracted_feature_iter80000/img_fea.mat')
img_fea = exp(x);

load('../extracted_feature_iter80000/txt_fea.mat')
txt_fea = exp(x);

load('../extracted_feature_iter80000/aud_fea.mat')
aud_fea = exp(x);

load('../extracted_feature_iter80000/vid_fea.mat')
vid_fea = exp(x);

load('../extracted_feature_iter80000/td_fea.mat')
td_fea = exp(x);


load('./listFile/test_img_txt_lab.mat');
img_lab = test_lab;
txt_lab = test_lab;
importdata('./listFile/test_aud.txt');
aud_lab = ans.data;
load('./listFile/test_vid_lab.mat');
vid_lab = test_lab;
importdata('./listFile/test_3d.txt');
td_lab = ans.data;

[a,b]=sort(img_lab);
img_fea=img_fea(b,:);
img_lab=img_lab(b);


[a,b]=sort(txt_lab);
txt_fea=txt_fea(b,:);
txt_lab=txt_lab(b);

[a,b]=sort(aud_lab);
aud_fea=aud_fea(b,:);
aud_lab=aud_lab(b);

[a,b]=sort(vid_lab);
vid_fea=vid_fea(b,:);
vid_lab=vid_lab(b);


[a,b]=sort(td_lab);
td_fea=td_fea(b,:);
td_lab=td_lab(b);

te_n_I = 8000;
te_n_T = 8000;
te_n_A = 2000;
te_n_V = 1000;
te_n_TD = 400;

D = pdist([img_fea; txt_fea; aud_fea; vid_fea; td_fea],'Cos');
W2 = -squareform(D);

W = W+W2;

WIA = W(1:te_n_I,:);
WTA = W(te_n_I+1:te_n_I+te_n_T,:);
WAAA = W(te_n_I+te_n_T+1:te_n_I+te_n_T+te_n_A,:);
WVA = W(te_n_I+te_n_T+te_n_A+1:te_n_I+te_n_T+te_n_A+te_n_V,:);
WTDA = W(te_n_I+te_n_T+te_n_A+te_n_V+1:end,:);

WII = W(1:te_n_I,1:te_n_I);
WTT = W(te_n_I+1:te_n_I+te_n_T,te_n_I+1:te_n_I+te_n_T);
WAA = W(te_n_I+te_n_T+1:te_n_I+te_n_T+te_n_A,te_n_I+te_n_T+1:te_n_I+te_n_T+te_n_A);
WVV = W(te_n_I+te_n_T+te_n_A+1:te_n_I+te_n_T+te_n_A+te_n_V,te_n_I+te_n_T+te_n_A+1:te_n_I+te_n_T+te_n_A+te_n_V);
WTDTD = W(te_n_I+te_n_T+te_n_A+te_n_V+1:end,te_n_I+te_n_T+te_n_A+te_n_V+1:end);

WIT = W(1:te_n_I,te_n_I+1:te_n_I+te_n_T);
WTI = W(te_n_I+1:te_n_I+te_n_T,1:te_n_I);


[mapIT, prIQ_IT, mapICategory_IT, ap_IT] = QryonTestBi(WIA, img_lab, [img_lab;txt_lab;aud_lab;vid_lab;td_lab]);
disp(['Image->All Query MAP: ' num2str(mapIT)]);
[mapIT, prIQ_IT, mapICategory_IT, ap_IT] = QryonTestBi(WTA, txt_lab, [img_lab;txt_lab;aud_lab;vid_lab;td_lab]);
disp(['Text->All Query MAP: ' num2str(mapIT)]);
[mapIT, prIQ_IT, mapICategory_IT, ap_IT] = QryonTestBi(WAAA, aud_lab, [img_lab;txt_lab;aud_lab;vid_lab;td_lab]);
disp(['Audio->All Query MAP: ' num2str(mapIT)]);
[mapIT, prIQ_IT, mapICategory_IT, ap_IT] = QryonTestBi(WVA, vid_lab, [img_lab;txt_lab;aud_lab;vid_lab;td_lab]);
disp(['Video->All Query MAP: ' num2str(mapIT)]);
[mapIT, prIQ_IT, mapICategory_IT, ap_IT] = QryonTestBi(WTDA, td_lab, [img_lab;txt_lab;aud_lab;vid_lab;td_lab]);
disp(['ThreeD->All Query MAP: ' num2str(mapIT)]);
[mapIT, prIQ_IT, mapICategory_IT, ap_IT] = QryonTestBi(WIT, img_lab, txt_lab);
disp(['image->text Query MAP: ' num2str(mapIT)]);
[mapIT, prIQ_IT, mapICategory_IT, ap_IT] = QryonTestBi(WTI, txt_lab, img_lab);
disp(['text->image Query MAP: ' num2str(mapIT)]);
