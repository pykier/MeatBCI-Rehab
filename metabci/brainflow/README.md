# A development note of *Brainflow*

## 1. Updating note

Now, *brainflow* is able to support more kinds of devices by:

* Native support Neuroscan and Neurcale by inheriting and re-written the BaseAmplifier class.
* Indirect support devices, such as [EGI](https://www.egi.com), [g.tec](https://www.gtec.at/) and [BioSemi](https://www.biosemi.com), by implementing a communication protocol between MetaBCI and [Lab streaming layer](https://github.com/sccn/labstreaminglayer). Due to the LSL provides many specific apps for retrieving data from different devices in a unified format, the MetaBCI could get and process online data from these devices by using LSL.
* For meeting the requirements of recording event triggers without hardware like Serial and LPT. We provide a feasible way to writing events triggers (also known as markers). Users can use it to write markers just like writing hardware triggers.

## 2. How to use

* You need to check [this website](https://labstreaminglayer.readthedocs.io/info/supported_devices.html) and download the LSL app for your device.
* Write some code to instantiate the LSLapp class.
* The remaining steps are similar with demo scripts [here]()

## 3. Suggestion and limitation

* If it is possible, **TRY TO USE HARDWARE EVENT TRIGGER**. As far as I know, the hardware event triggers enable more precise synchronization of events with the device data streams, in most of the situation. So if you had a trigger box or any other similar devices, try to use it without applying the software event trigger.
* If you owned a physical device, and also able to acquire the communication protocol code or instructions for getting data from the devices. We strongly suggest that you can try to inherit and re-write the BaseAmplifier class for help MetaBCI to native more device.
* Our lab (TUNERL) owned an EGI device, the metabci team planned to support EGI first.
* Considering the differences among different devices for transfering the event trigger.
    **YOU MUST BE VERY CAREFUL** to determine wethher the data stream reading
    from the LSL apps contains a event channel. For example, the neuroscan
    synamp II will append a extra event channel to the raw data channel.
    Because we do not have chance to test each device that LSL supported, so
    please modify this class before using with your own condition.
<!-- 
    Brainflow 开发说明
1. 更新说明

目前，brainflow 已经能够通过以下方式支持更多类型的设备：

通过继承并重写 BaseAmplifier 类，原生支持 Neuroscan 和 Neuracle 设备。
通过在 MetaBCI 和 Lab Streaming Layer, LSL 之间实现通信协议，间接支持一些设备，例如 EGI、g.tec 和 BioSemi。由于 LSL 为不同设备提供了许多专用应用程序，可以用统一格式获取数据，因此 MetaBCI 可以通过 LSL 获取并处理这些设备的在线数据。
为了满足在没有串口、LPT 并口等硬件触发设备时记录事件触发的需求，系统提供了一种可行的软件事件触发写入方式。用户可以像写入硬件触发一样写入事件标记，也称为 markers。
2. 使用方法
需要查看这个网站，并下载适用于自己设备的 LSL 应用程序。
编写代码实例化 LSLapp 类。
其余步骤与这里的 demo 脚本类似。
3. 建议与限制
如果条件允许，尽量使用硬件事件触发。据我所知，在大多数情况下，硬件事件触发能够实现事件与设备数据流之间更加精确的同步。因此，如果你有 trigger box 或其他类似设备，建议优先使用硬件触发，而不是软件事件触发。
如果你拥有实际设备，并且能够获得该设备的通信协议代码或数据读取说明，强烈建议你尝试继承并重写 BaseAmplifier 类，以帮助 MetaBCI 原生支持更多设备。
我们实验室 TUNERL 拥有一台 EGI 设备，因此 MetaBCI 团队计划优先支持 EGI。
由于不同设备传输事件触发的方式存在差异，在使用时必须非常谨慎地判断从 LSL 应用读取的数据流中是否包含事件通道。例如，Neuroscan SynAmps II 会在原始数据通道后额外添加一个事件通道。由于我们没有机会测试 LSL 支持的每一种设备，因此在实际用于你自己的设备之前，请根据具体情况修改这个类。 -->