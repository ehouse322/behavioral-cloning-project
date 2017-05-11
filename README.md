n this project, I collected data by controlling a simulated car through a track. I then trained a neural network on this model to predict steering wheel angles based on the images collected by the car as it goes through the track. The end result is that the car is able to successfully travel through the track autonomously.

If you would like to test this out, you can first download Udacity's car simulator here:
    - [linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)
    - [mac](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip)
    - [windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)

After that, run:

 `python drive.py model.h5`

While this is running, open up the car simulator, select the first track, and click autonomous mode! 
