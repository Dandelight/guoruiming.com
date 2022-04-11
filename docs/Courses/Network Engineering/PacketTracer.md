## 实验目标

掌握交换机基本信息的配置管理。

## 技术原理

交换机的管理方式基本分为两种：**带内管理**和**带外管理**。

通过交换机的 Console 端口管理交换机属于带外管理。这种管理方式不占用交换机的网络端口，第一次配置交换机必须利用 Console 端口进行配置。

通过Telnet、拨号等方式登陆终端属于带内管理。

交换机的命令行操作模式主要包括：


* 用户模式 `Switch>`
* 特权模式 `Switch#`
* 全局配置模式 `Switch(config)#`
* 端口模式 `Switch(config-if)#`


交换机常用的配置命令行

模式切换指令

enable 进入特权模式（一般简写为 en）

config t 进入全局配置模式

interface fa 0/1 进入交换机某个端口视图模式

exit 返回到上级模式

end 从全局以下模式返回到特权模式

快捷指令

帮助信息(如? 、co?、copy?)

命令简写(如 en 的完整命令为 enable)

命令自动补全(Tab)

快捷键(ctrl+c 中断测试,ctrl+z 退回到特权视图)

reload 重启(在特权模式下)

hostname X 修改交换机名称(在全局配置模式下)

端口配置指令

speed,duplex 配置交换机端口参数

show version 查看交换机版本信息

show running-config 查看当前生效的配置信息

show startup-config 查看保存在 NVRAM 中的启动配置信息

show interface 查看端口信息

show mac-address-table 查看交换机的 MAC 地址

选择某个端口 Switch(config)# interface type mod/port

(type：端口类型，通常有 ethernet、Fastethernet、Gigabitethernet；

mod：端口所在的模块；

port：在该模块中的编号；）:Switch(config)# interface fa 0/1;

选择多个端口 Switch(config)#interface type mod/startport-endport

如：Switch(config)# interface interface fa 0/1-5 //选择端口 fa 0/1 ~ fa 0/5

Switch(config-if)#speed [10/100/auto] 设置端口通信速度

Switch(config-if)#duplex [half/full/auto] 设置端口单双工模式

若交换机设置为 auto 以外的具体速度，此时应注意保证通信双方也要有相同的设置值。

注意事项：在配置交换机时，要注意交换机端口的单双工模式的匹配，如果链路一端设置的是全双工，另一端是自动协商，则会造成响应差和高出错率，丢包现象会很严重。通常两端设置为相同的模式。

密码设置指令

设置进入特权模式的密码

Switch(config)# enable password **\*\***

通过 console 端口连接设备及 Telnet 远程登录时所需的密码；

Switch(config)# line console 0 表示配置控制台线路，0 是控制台的线路编号。

Switch(config-line)# login 用于打开登录认证功能。

Switch(config-line)# password 5ijsj 设置进入控制台访问的密码

若交换机设置为 auto 以外的具体速度，此时应注意保证通信双方也要有相同的设置值。

注意事项：在配置交换机时，要注意交换机端口的单双工模式的匹配，如果链路一端设置的是全双工，另一端是自动协商，则会造成响应差和高出错率，丢包现象会很严重。通常两端设置为相同的模式。

实验设备

Switch_2960 1 台；PC 1 台；直通线和配置线；

[^youzi]: https://blog.csdn.net/gengkui9897/article/details/85109962
