import image_clean

# image_clean.denoise_predict("images/denoise/*.png", "output/denoise")
# image_clean.deblur_predict("images/deblur/*.png", "output/deblur")

# demo
# image_clean.denoise_predict("images/noisy15/*.png", "output/demo/denoise")
image_clean.denoise_add_predict("images/McMaster/*.tif", "output/demo/denoise")
# image_clean.deblur_predict("images/GoPro_Deblur/G*_11_[0-1]*-000*0[0-1].png", "output/demo/deblur")
