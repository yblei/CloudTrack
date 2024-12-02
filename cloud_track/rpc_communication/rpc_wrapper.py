import requests
from tinyrpc import RPCClient
from tinyrpc.protocols.msgpackrpc import MSGPACKRPCProtocol
from tinyrpc.transports.http import HttpPostClientTransport

from cloud_track.foundation_model_wrappers import WrapperBase
from cloud_track.utils.flow_control import PerformanceTimer

from .utils import deserialize_results, serialize_args, serialize_kwargs


class RpcWrapper(WrapperBase):
    """Forwards inference requests to the RPC server."""

    def __init__(self, ip: str, port: int) -> None:

        if not "http://" in ip:
            raise ValueError(
                "Please provide the full URL, including the protocol"
                " e.g. http://."
            )

        # fix to reuse session in tinyrpc. If not we run out of ports.
        endpoint = f"{ip}:{port}"
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=50)
        session.mount(endpoint, adapter)
        transport = HttpPostClientTransport(
            endpoint,
            post_method=session.post,
        )

        rpc_client = RPCClient(MSGPACKRPCProtocol(), transport)
        self.backend = rpc_client.get_proxy()

        self.encoding_timer = PerformanceTimer()
        self.decoding_timer = PerformanceTimer()

    def run_inference(self, *args, **kwargs):
        # serialize the arguments
        with self.encoding_timer:
            args = serialize_args(args)
            kwargs = serialize_kwargs(kwargs)

        # call the RPC server
        result = self.backend.run_inference(*args, **kwargs)

        # deserialize the result
        with self.decoding_timer:
            result = deserialize_results(result)

        return result

    def get_stats(self):
        return f"RPC Statistics:\nEncoding Time: {self.encoding_timer}s\nDecoding Time: {self.decoding_timer}s"


# from https://github.com/raiden-network/raiden/pull/350/commits/e3b9a32aa95cd925ad9118a011f51a88e50e9b1b
def patch_send_message(client, pool_maxsize=50):
    """Monkey patch fix for issue #253. This makes the underlying `tinyrpc`
    transport class use a `requests.session` instead of regenerating sessions
    for each request.
    See also: https://github.com/mbr/tinyrpc/pull/31 for a proposed upstream
    fix.
    Args:
        client (pyethapp.rpc_client.JSONRPCClient): the instance to patch
        pool_maxsize: the maximum poolsize to be used by the `requests.Session()`
    """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=pool_maxsize)
    session.mount(client.transport.endpoint, adapter)

    def send_message(message, expect_reply=True):
        if not isinstance(message, str):
            raise TypeError("str expected")

        r = session.post(
            client.transport.endpoint,
            data=message,
            **client.transport.request_kwargs,
        )

        if expect_reply:
            return r.content

    client.transport.send_message = send_message
