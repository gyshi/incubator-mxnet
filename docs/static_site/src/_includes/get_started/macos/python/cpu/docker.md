Docker images with *MXNet* are available at [DockerHub](https://hub.docker.com/r/mxnet/).

**Step 1** Install Docker on your machine by following the [docker installation
instructions](https://docs.docker.com/docker-for-mac/install/#install-and-run-docker-for-mac).

*Note* - You can install Community Edition (CE) to get started with *MXNet*.

**Step 2** Pull the MXNet docker image.

{% highlight bash %}
$ docker pull mxnet/python
{% endhighlight %}

You can list docker images to see if mxnet/python docker image pull was successful.

{% highlight bash %}
$ docker images

REPOSITORY TAG IMAGE ID CREATED SIZE
mxnet/python latest 00d026968b3c 3 weeks ago 1.41 GB
{% endhighlight %}

Using the latest MXNet with [Intel MKL-DNN](https://github.com/intel/mkl-dnn) is
recommended for the
fastest inference speeds with MXNet.

{% highlight bash %}
$ docker pull mxnet/python:1.3.0_cpu_mkl # Use sudo if you skip Step 2
$ docker images # Use sudo if you skip Step 2

REPOSITORY TAG IMAGE ID CREATED SIZE
mxnet/python 1.3.0_cpu_mkl deaf9bf61d29 4 days ago 678 MB
{% endhighlight %}

**Step 4** <a href="validate_mxnet.html">Validate the installation</a>.