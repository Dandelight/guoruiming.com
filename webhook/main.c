#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>
#include <sys/epoll.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/stat.h>
#include <signal.h>
#include <ctype.h>
#include <assert.h>
#define MAXSIZE 2000

struct epoll_event g_Events[MAXSIZE];
char *blog_root;

void check_ret(int ret, char *msg) {
    int exitcode = -1;
    if(ret == -1) {
        perror(msg);
        exit(exitcode);
    }
}

int init_listen_fd(int port, int epfd) {
    int lfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

    struct sockaddr_in serv;
    memset(&serv, 0x0, sizeof(serv));
    serv.sin_family = AF_INET;
    serv.sin_port = htons(port);
    serv.sin_addr.s_addr = INADDR_ANY;

    int flag = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(flag));
    int ret = bind(lfd, (struct sockaddr *)&serv, sizeof(serv));
    check_ret(ret, "Bind error");

    ret = listen(lfd, 64);
    check_ret(ret, "Listen error");

    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.fd = lfd;
    ret = epoll_ctl(epfd, EPOLL_CTL_ADD, lfd, &ev);
    check_ret(ret, "epoll_ctl add error");

    return lfd;
}

void do_accept(int lfd, int epfd) {
    struct sockaddr_in client;
    socklen_t len = sizeof(client);
    int cfd = accept(lfd, (struct sockaddr *)&client, &len);
    check_ret(cfd, "Accept error");

    // 打印客户端信息
    char ip[64] = {0};
    // 设置cfd为非阻塞
    int flag = fcntl(cfd, F_GETFL);
    flag |= O_NONBLOCK;
    fcntl(cfd, F_SETFL, flag);
    // 得到的新节点挂到epoll树上
    struct epoll_event ev;
    ev.data.fd = cfd;
    // 边沿非阻塞模式
    ev.events = EPOLLIN | EPOLLET;
    int ret = epoll_ctl(epfd, EPOLL_CTL_ADD, cfd, &ev);
    check_ret(ret, "epoll_ctl add cfd error");
}

void disconnect(int cfd, int epfd)
{
    int ret = epoll_ctl(epfd, EPOLL_CTL_DEL, cfd, NULL);
    check_ret(ret, "epoll_ctl del cfd error");
    close(cfd);
}

void update_blog() {
    int ret = chdir(blog_root);
    check_ret(ret, "chdir error");
    system("bash ./deploy.sh");
}

void do_read(int cfd, int epfd) {
    char c;
    while(recv(cfd, &c, 1, 0) > 0) {
        putchar(c);
    }
    update_blog();
    disconnect(cfd, epfd);
}

void run_epoll(int port) {
    int epfd = epoll_create(MAXSIZE);
    int lfd = init_listen_fd(port, epfd);

    while(1) {
        int ret = epoll_wait(epfd, g_Events, MAXSIZE, -1);
        check_ret(ret, "Epoll wait error");

        for(int i = 0; i < ret; ++ i) {
            struct epoll_event *pev = &g_Events[i];
            if(!(pev->events & EPOLLIN)) {
                continue;
            }
            if(pev->data.fd == lfd) {
                do_accept(lfd, epfd);
            }
            else {
                do_read(pev->data.fd, epfd);
            }
        }
    }
}

int main(int argc, const char *argv[])
{
  int port;
  assert(getenv("HOOK_PORT"));
  sscanf(getenv("HOOK_PORT"), "%d", &port);

  blog_root = getenv("BLOG_ROOT");
  assert(blog_root);

  run_epoll(port);
  return 0;
}
