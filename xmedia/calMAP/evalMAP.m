clear

load('../extracted_feature/img_fea.mat');
img_fea = 10.^x;
load('../extracted_feature/txt_fea.mat');
txt_fea = 10.^x;
load('../extracted_feature/aud_fea.mat');
aud_fea = 10.^x;
load('../extracted_feature/vid_fea.mat');
vid_fea = 10.^x;
load('../extracted_feature/td_fea.mat');
td_fea = 10.^x;

importdata('./listFile/test_img.txt');
img_lab = ans.data;
importdata('./listFile/test_txt.txt');
txt_lab = ans.data;
importdata('./listFile/test_aud.txt');
aud_lab = ans.data;
importdata('./listFile/test_vid.txt');
vid_lab = ans.data;
importdata('./listFile/test_3d.txt');
td_lab = ans.data;

[a,b]=sort(img_lab);
bb=reshape(b,50,20);
tb=bb(11:end,:);
tb=reshape(tb,800,1);
img_fea=img_fea(tb,:);
img_lab=img_lab(tb);

[a,b]=sort(txt_lab);
bb=reshape(b,50,20);
tb=bb(11:end,:);
tb=reshape(tb,800,1);
txt_fea=txt_fea(tb,:);
txt_lab=txt_lab(tb);

[a,b]=sort(aud_lab);
bb=reshape(b,10,20);
tb=bb(3:end,:);
tb=reshape(tb,160,1);
aud_fea=aud_fea(tb,:);
aud_lab=aud_lab(tb);

[a,b]=sort(vid_lab);
bb=reshape(b,5,20);
tb=bb(2:end,:);
tb=reshape(tb,80,1);
vid_fea=vid_fea(tb,:);
vid_lab=vid_lab(tb);

[a,b]=sort(td_lab);
bb=reshape(b,5,20);
tb=bb(2:end,:);
tb=reshape(tb,80,1);
td_fea=td_fea(tb,:);
td_lab=td_lab(tb);

te_n_I = 800;
te_n_T = 800;
te_n_A = 160;
te_n_V = 80;
te_n_TD = 80;

D = pdist([img_fea; txt_fea; aud_fea; vid_fea; td_fea],'Cos');
W = -squareform(D);

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
disp(['Image->Text Query MAP: ' num2str(mapIT)]);
[mapIT, prIQ_IT, mapICategory_IT, ap_IT] = QryonTestBi(WTI, txt_lab, img_lab);
disp(['Text->Image Query MAP: ' num2str(mapIT)]);
