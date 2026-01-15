## Nori Version 2

Nori is a simple ray tracer written in C++. It runs on Windows, Linux, and
ac OS and provides basic functionality on which renderers can be built.
This software is kindly loaned by Wenzel Jakob, who developed Nori originally
for CS6630 and developed this version with his students to support the
Advanced Computer Graphics course at EPFL.

### Handing in via GitHub URL

When you hand in CS5630 assignments, please do so by uploading a URL that points to a 
specific commit in the Git repository on the Cornell COECIS GitHub instance that you
created by forking our class repository.

1. Open your repository on the GitHub web interface.
2. Navigate to your report: `results` > a`<assignment-number>` > `report.html`
3. Press the `y` key on your keyboard. Note that the URL now has the form:
```https://github.coecis.cornell.edu/<your-netid>/nori-26sp/blob/<commit-id>/results/<assignment-name>/report.html```
Copy this URL and submit it on Canvas.

Note that the following kinds of URLs do not uniquely identify a specific state, and are not good submissions for CS5630:

1. A link to a branch or tag: `https://github.coecis.cornell.edu/<your-netid>/nori-26sp/blob/master/results/a0/report.html`
2. A link to the base repository: `https://github.coecis.cornell.edu/cs6630/nori-26sp/blob/17f31bff65f3476ec3f94685b68ddafeca92b431/results/a0/report.html`
3. A link to a specific state that you later overwrote so that it results in a 404. You need to be extra-careful to never rewrite the repository's history after a submission, at the risk of losing your ability to demonstrate what code you had before the deadline. (This is a fairly advanced Git move that you are not likely to do without knowing you are doing it!)


### Known Issues
There is a known issue with the NanoGUI version that Nori uses: on Linux systems with an integrated Intel GPU, a bug in the Mesa graphics drivers causes the GUI to freeze on startup. A workaround is to temporarily switch to an older Mesa driver to run Nori. This can be done by running
```
export MESA_LOADER_DRIVER_OVERRIDE=i965
```
