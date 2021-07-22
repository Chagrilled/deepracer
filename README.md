# deepracer

Scripts to calculate action space and optimal racelines

I used Anaconda python to install stuff, and run through anaconda navigator seemed to work well, but was weird

Track data comes from: https://github.com/aws-deepracer-community/deepracer-simapp/tree/master/bundle/deepracer_simulation_environment/share/deepracer_simulation_environment/routes

This could also be a good repo: https://github.com/cdthompson/deepracer-training-2019

Change the track name in linecalc and run:

```bash
python linecalc.py # Calc the raceline

```

Then put the output filename into action_space.py and run it

Then you've got the optimal speeds/coords to copy into the array in the reward func

cdthompson's uses the raceline, and Capstone's uses the action-space augmented data.

Optimal time for Empire according to our raceline is 9.849719547766956 s