from adaptive import *


####################################################
########### MAIN LOOP OVER SCENE POINTS ############
####################################################

tfront = np.load("../data/leaf.npy")
frm_img = np.zeros((128, 128))
ad_img = np.zeros((128, 128))
cc = 0
exposure = 1000000
for i in range(128):
    for j in range(128):
        cc+=1
        bkg, sig, td = get_trans(tfront[i, j])
        Ms = gen_Ms(bkg, sig)
        
        # Setting depth prior
        prior = np.ones(500)/500

        # Adaptive Gating Iteration
        log_pdf, elapsed, N, D = adapt_shift(prior, tfront[i,j], exposure, Ms, dead_time = 810, mode = 0)
        d = np.argmax(log_pdf)
        if d > 490 or d < 10:
            ad_img[i, j] = d
        else:
            ctrans = np.log(1/(1-(N[d-4:d+5]/D[d-4:d+5])))
            ctrans = ctrans - np.min(ctrans)
            bias = np.sum(ctrans*np.arange(-4,5))/np.sum(ctrans)
            ad_img[i, j] = d + bias
        
        # Free-Running Mode Iteration
        log_pdf, elapsed, N, D = adapt_shift(prior, tfront[i,j], exposure, Ms, dead_time = 810, mode = 1)
        d = np.argmax(log_pdf)
        if d > 490 or d < 10:
            frm_img[i, j] = d
        else:
            ctrans = np.log(1/(1-(N[d-4:d+5]/D[d-4:d+5])))
            ctrans = ctrans - np.min(ctrans)
            bias = np.sum(ctrans*np.arange(-4,5))/np.sum(ctrans)
            frm_img[i, j] = d + bias


####################################################
################## VISUALISATIONS ##################
####################################################

# Image clean up, firefly errors are clamped for better presentation
ad_img_thresh[ad_img < 67.13844688839994] = 67.13844688839994
ad_img_thresh[ad_img > 83.71508734868411] = 83.71508734868411
ad_out = ad_img_thresh.T[-20:3:-1,110:13:-1]
ad_out = np.delete(ad_out, 74, 1) 
frm_img_thresh[frm_img < 67.13844688839994] = 67.13844688839994
frm_img_thresh[frm_img > 83.71508734868411] = 83.71508734868411
frm_out = ad_img_thresh.T[-20:3:-1,110:13:-1]
frm_out = np.delete(frm_out, 74, 1) 

plt.matshow(ad_out, cmap = "jet_r")
plt.axis('off')
plt.show()
plt.matshow(frm_out, cmap = "jet_r")
plt.axis('off')
plt.show()