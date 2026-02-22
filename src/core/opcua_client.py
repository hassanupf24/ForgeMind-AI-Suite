"""
ForgeMind AI Suite — OPC-UA Client
Async OPC-UA client for reading industrial asset data nodes.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from asyncua import Client, ua

from src.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class OPCUAClient:
    """Async OPC-UA client for industrial asset data acquisition."""

    def __init__(self, server_url: Optional[str] = None) -> None:
        self._url = server_url or settings.opcua_server_url
        self._client: Optional[Client] = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to OPC-UA server."""
        self._client = Client(self._url)
        self._client.set_security_string(
            f"{settings.opcua_security_policy},SignAndEncrypt"
        )
        try:
            await self._client.connect()
            self._connected = True
            logger.info("OPC-UA connected to %s", self._url)
        except Exception as e:
            self._connected = False
            logger.error("OPC-UA connection failed: %s", e)
            raise

    async def disconnect(self) -> None:
        """Disconnect from OPC-UA server."""
        if self._client:
            await self._client.disconnect()
            self._connected = False
            logger.info("OPC-UA disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def read_node(self, node_id: str) -> Any:
        """Read a single node value from OPC-UA server.

        Args:
            node_id: Node identifier, e.g. 'ns=2;s=Machine.001.Vibration'
        """
        if not self._client or not self._connected:
            raise RuntimeError("OPC-UA client not connected")
        node = self._client.get_node(node_id)
        value = await node.read_value()
        return value

    async def read_multiple_nodes(self, node_ids: list[str]) -> dict[str, Any]:
        """Read multiple node values in a single call."""
        if not self._client or not self._connected:
            raise RuntimeError("OPC-UA client not connected")

        results = {}
        nodes = [self._client.get_node(nid) for nid in node_ids]
        values = await self._client.read_values(nodes)

        for nid, val in zip(node_ids, values):
            results[nid] = val

        return results

    async def read_machine_telemetry(self, machine_id: str) -> dict[str, float]:
        """Read standard telemetry nodes for a machine.

        Reads: Vibration, Temperature, Pressure, RPM, Current
        Node pattern: ns=2;s=Machine.{id}.{param}
        """
        params = ["Vibration", "Temperature", "Pressure", "RPM", "Current"]
        node_ids = [f"ns=2;s=Machine.{machine_id}.{p}" for p in params]

        try:
            raw = await self.read_multiple_nodes(node_ids)
            return {
                "vibration_ms2": raw.get(node_ids[0], 0.0),
                "temperature_C": raw.get(node_ids[1], 0.0),
                "pressure_bar": raw.get(node_ids[2], 0.0),
                "rpm": raw.get(node_ids[3], 0.0),
                "current_A": raw.get(node_ids[4], 0.0),
            }
        except Exception as e:
            logger.error("Failed to read telemetry for machine %s: %s", machine_id, e)
            return {}

    async def subscribe_data_change(
        self,
        node_ids: list[str],
        handler,
        period: int = 1000,
    ):
        """Subscribe to data change events on OPC-UA nodes.

        Args:
            node_ids: Nodes to monitor
            handler: Async callback for data changes
            period: Subscription period in milliseconds
        """
        if not self._client or not self._connected:
            raise RuntimeError("OPC-UA client not connected")

        subscription = await self._client.create_subscription(period, handler)
        nodes = [self._client.get_node(nid) for nid in node_ids]
        await subscription.subscribe_data_change(nodes)
        logger.info("OPC-UA subscription created for %d nodes", len(node_ids))
        return subscription


# Singleton instance
opcua_client = OPCUAClient()
