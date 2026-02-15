"""
通信接口抽象模块
支持WebSocket和TCP两种通信方式
"""

import logging
import threading
import json
import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional, Callable

logger = logging.getLogger(__name__)


class CommunicationInterface(ABC):
    """通信接口抽象基类"""
    
    @abstractmethod
    def start(self) -> bool:
        """启动通信服务"""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """停止通信服务"""
        pass
    
    @abstractmethod
    def send_tag(self, tag: str) -> bool:
        """
        发送意图tag
        
        参数:
            tag: 意图tag
            
        返回:
            是否发送成功
        """
        pass
    
    @abstractmethod
    def set_control_callback(self, callback: Callable[[bool], None]) -> None:
        """
        设置控制回调函数
        
        参数:
            callback: 回调函数，接收一个bool参数（是否启用音频播放）
        """
        pass


class WebSocketCommunication(CommunicationInterface):
    """WebSocket通信实现"""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.url = f"ws://{host}:{port}"
        self.websocket_connection = None
        self.websocket_loop = None
        self.websocket_lock = threading.Lock()
        self.control_callback = None
        self.client_thread = None
        self.running = False
    
    def start(self) -> bool:
        """启动WebSocket客户端"""
        if self.running:
            logger.warning("WebSocket客户端已在运行")
            return False
        
        self.running = True
        self.client_thread = threading.Thread(target=self._client_thread, daemon=True)
        self.client_thread.start()
        logger.info("WebSocket客户端线程已启动")
        return True
    
    def stop(self) -> None:
        """停止WebSocket客户端"""
        self.running = False
        with self.websocket_lock:
            if self.websocket_connection:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.websocket_connection.close(),
                        self.websocket_loop
                    )
                except:
                    pass
                self.websocket_connection = None
                self.websocket_loop = None
    
    def send_tag(self, tag: str) -> bool:
        """发送tag到WebSocket服务器"""
        if not tag:
            return False
        
        with self.websocket_lock:
            if self.websocket_connection is None or self.websocket_loop is None:
                logger.warning("WebSocket 未连接，无法发送 tag")
                return False
            
            try:
                asyncio.run_coroutine_threadsafe(
                    self.websocket_connection.send(tag),
                    self.websocket_loop
                )
                logger.info(f"已发送 tag {tag} 给 WebSocket 服务器")
                return True
            except Exception as e:
                logger.warning(f"发送 tag 给 WebSocket 服务器失败: {e}")
                return False
    
    def set_control_callback(self, callback: Callable[[bool], None]) -> None:
        """设置控制回调函数"""
        self.control_callback = callback
    
    def _handle_control_message(self, message: str) -> Optional[dict]:
        """处理控制消息"""
        try:
            try:
                msg_dict = json.loads(message)
                msg_type = msg_dict.get("type", "")
                
                if msg_type == "control":
                    enable = msg_dict.get("enable_audio", True)
                    if self.control_callback:
                        self.control_callback(bool(enable))
                    return {
                        "type": "control_response",
                        "status": "ok",
                        "enable_audio": enable
                    }
                elif msg_type == "ping":
                    return {"type": "pong"}
            except json.JSONDecodeError:
                message_lower = message.strip().lower()
                if message_lower in ["enable", "true", "1", "on"]:
                    if self.control_callback:
                        self.control_callback(True)
                    return {"status": "ok", "message": "audio enabled"}
                elif message_lower in ["disable", "false", "0", "off"]:
                    if self.control_callback:
                        self.control_callback(False)
                    return {"status": "ok", "message": "audio disabled"}
        except Exception as e:
            logger.error(f"处理 WebSocket 控制消息时出错: {e}")
        return None
    
    def _client_thread(self) -> None:
        """WebSocket客户端线程"""
        import websockets
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._client_loop())
        except Exception as e:
            logger.error(f"WebSocket 客户端线程出错: {e}")
        finally:
            loop.close()
    
    async def _client_loop(self) -> None:
        """WebSocket客户端循环"""
        import websockets
        
        while self.running:
            try:
                logger.info(f"正在连接到 WebSocket 服务器: {self.url}")
                async with websockets.connect(self.url) as websocket:
                    logger.info(f"✓ 已连接到 WebSocket 服务器: {self.url}")
                    
                    with self.websocket_lock:
                        self.websocket_connection = websocket
                        self.websocket_loop = asyncio.get_event_loop()
                    
                    try:
                        async for message in websocket:
                            response = self._handle_control_message(message)
                            if response and isinstance(response, dict):
                                try:
                                    response_str = json.dumps(response, ensure_ascii=False)
                                    await websocket.send(response_str)
                                    logger.debug(f"已发送响应给服务器: {response_str}")
                                except Exception as e:
                                    logger.warning(f"发送响应给服务器失败: {e}")
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket 连接已关闭")
                    except Exception as e:
                        logger.error(f"接收 WebSocket 消息时出错: {e}")
                    finally:
                        with self.websocket_lock:
                            self.websocket_connection = None
                            self.websocket_loop = None
            except Exception as e:
                logger.error(f"WebSocket 连接失败: {e}")
                with self.websocket_lock:
                    self.websocket_connection = None
                    self.websocket_loop = None
                
                if self.running:
                    logger.info("5 秒后重试连接...")
                    await asyncio.sleep(5)


class TCPCommunication(CommunicationInterface):
    """TCP通信实现"""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.server_socket = None
        self.server_thread = None
        self.connected_clients: List = []
        self.clients_lock = threading.Lock()
        self.control_callback = None
        self.running = False
    
    def start(self) -> bool:
        """启动TCP服务器"""
        if self.running:
            logger.warning("TCP服务器已在运行")
            return False
        
        self.running = True
        self.server_thread = threading.Thread(target=self._server_thread, daemon=True)
        self.server_thread.start()
        logger.info("TCP服务器线程已启动")
        return True
    
    def stop(self) -> None:
        """停止TCP服务器"""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            self.server_socket = None
    
    def send_tag(self, tag: str) -> bool:
        """发送tag到所有连接的客户端"""
        if not tag:
            return False
        
        import socket
        
        message = json.dumps({"type": "tag", "data": tag}, ensure_ascii=False)
        message_bytes = (message + "\n").encode('utf-8')
        
        with self.clients_lock:
            disconnected_clients = []
            for client_socket in self.connected_clients:
                try:
                    client_socket.sendall(message_bytes)
                    logger.debug(f"已发送 tag {tag} 给客户端")
                except Exception as e:
                    logger.warning(f"发送 tag 给客户端失败: {e}")
                    disconnected_clients.append(client_socket)
            
            for client in disconnected_clients:
                if client in self.connected_clients:
                    self.connected_clients.remove(client)
                    try:
                        client.close()
                    except:
                        pass
        
        return True
    
    def set_control_callback(self, callback: Callable[[bool], None]) -> None:
        """设置控制回调函数"""
        self.control_callback = callback
    
    def _handle_client(self, client_socket, address) -> None:
        """处理客户端连接"""
        logger.info(f"客户端连接: {address}")
        
        try:
            while self.running:
                data = client_socket.recv(1024)
                if not data:
                    break
                
                try:
                    message = data.decode('utf-8').strip()
                    logger.info(f"收到客户端消息: {message}")
                    
                    try:
                        msg_dict = json.loads(message)
                        msg_type = msg_dict.get("type", "")
                        
                        if msg_type == "control":
                            enable = msg_dict.get("enable_audio", True)
                            if self.control_callback:
                                self.control_callback(bool(enable))
                            
                            response = json.dumps({
                                "type": "control_response",
                                "status": "ok",
                                "enable_audio": enable
                            }, ensure_ascii=False)
                            client_socket.sendall((response + "\n").encode('utf-8'))
                        elif msg_type == "ping":
                            response = json.dumps({"type": "pong"}, ensure_ascii=False)
                            client_socket.sendall((response + "\n").encode('utf-8'))
                    except json.JSONDecodeError:
                        message_lower = message.lower()
                        if message_lower in ["enable", "true", "1", "on"]:
                            if self.control_callback:
                                self.control_callback(True)
                            client_socket.sendall(b"OK: audio enabled\n")
                        elif message_lower in ["disable", "false", "0", "off"]:
                            if self.control_callback:
                                self.control_callback(False)
                            client_socket.sendall(b"OK: audio disabled\n")
                        else:
                            client_socket.sendall(b"ERROR: unknown command\n")
                except UnicodeDecodeError:
                    logger.warning(f"无法解码客户端消息: {data}")
                    client_socket.sendall(b"ERROR: invalid encoding\n")
        except Exception as e:
            logger.error(f"处理客户端 {address} 时出错: {e}")
        finally:
            with self.clients_lock:
                if client_socket in self.connected_clients:
                    self.connected_clients.remove(client_socket)
            try:
                client_socket.close()
            except:
                pass
            logger.info(f"客户端断开连接: {address}")
    
    def _server_thread(self) -> None:
        """TCP服务器线程"""
        import socket
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            logger.info(f"TCP 服务器启动，监听 {self.host}:{self.port}")
            
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    
                    with self.clients_lock:
                        self.connected_clients.append(client_socket)
                    
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                except Exception as e:
                    if self.running:
                        logger.error(f"接受客户端连接时出错: {e}")
                    break
        except Exception as e:
            logger.error(f"TCP 服务器启动失败: {e}")
        finally:
            if self.server_socket:
                try:
                    self.server_socket.close()
                except:
                    pass


def create_communication_instance(comm_type: str, host: str, port: int) -> Optional[CommunicationInterface]:
    """
    创建通信实例
    
    参数:
        comm_type: 通信类型 ("websocket" 或 "tcp")
        host: 主机地址
        port: 端口号
        
    返回:
        通信实例，如果类型不支持则返回None
    """
    if comm_type.lower() == "websocket":
        return WebSocketCommunication(host, port)
    elif comm_type.lower() == "tcp":
        return TCPCommunication(host, port)
    else:
        logger.error(f"不支持的通信类型: {comm_type}")
        return None

