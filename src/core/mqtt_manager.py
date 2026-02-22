"""
ForgeMind AI Suite — MQTT Client Manager
Handles MQTT connections for IoT sensor data (smart meters, environmental sensors, PLC triggers).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

import paho.mqtt.client as mqtt

from src.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class MQTTManager:
    """Manages MQTT client connections for IoT sensor integration."""

    def __init__(self) -> None:
        self._client: Optional[mqtt.Client] = None
        self._handlers: dict[str, Callable[[str, dict[str, Any]], None]] = {}
        self._connected = False

    def _on_connect(
        self, client: mqtt.Client, userdata: Any, flags: dict, rc: int
    ) -> None:
        if rc == 0:
            self._connected = True
            logger.info("MQTT connected to %s:%d", settings.mqtt_broker_host, settings.mqtt_broker_port)
            # Re-subscribe on reconnect
            for topic in self._handlers:
                client.subscribe(topic, qos=1)
                logger.info("Subscribed to MQTT topic: %s", topic)
        else:
            logger.error("MQTT connection failed with code: %d", rc)

    def _on_disconnect(self, client: mqtt.Client, userdata: Any, rc: int) -> None:
        self._connected = False
        if rc != 0:
            logger.warning("MQTT unexpected disconnection (rc=%d), will auto-reconnect", rc)

    def _on_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            topic = msg.topic

            # Find matching handler (supports wildcards)
            for pattern, handler in self._handlers.items():
                if mqtt.topic_matches_sub(pattern, topic):
                    handler(topic, payload)
                    return

            logger.warning("No handler for MQTT topic: %s", topic)
        except json.JSONDecodeError:
            logger.error("Invalid JSON on MQTT topic %s: %s", msg.topic, msg.payload[:200])
        except Exception as e:
            logger.error("Error handling MQTT message: %s", e, exc_info=True)

    def connect(self) -> None:
        """Establish MQTT connection."""
        self._client = mqtt.Client(
            client_id=settings.mqtt_client_id,
            protocol=mqtt.MQTTv5,
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        if settings.mqtt_username:
            self._client.username_pw_set(settings.mqtt_username, settings.mqtt_password)

        self._client.reconnect_delay_set(min_delay=1, max_delay=60)
        self._client.connect_async(
            host=settings.mqtt_broker_host,
            port=settings.mqtt_broker_port,
            keepalive=60,
        )
        self._client.loop_start()
        logger.info("MQTT client connecting to %s:%d", settings.mqtt_broker_host, settings.mqtt_broker_port)

    def subscribe(
        self, topic: str, handler: Callable[[str, dict[str, Any]], None], qos: int = 1
    ) -> None:
        """Subscribe to an MQTT topic with a message handler."""
        self._handlers[topic] = handler
        if self._client and self._connected:
            self._client.subscribe(topic, qos=qos)
            logger.info("Subscribed to MQTT topic: %s", topic)

    def publish(
        self, topic: str, payload: dict[str, Any], qos: int = 1, retain: bool = False
    ) -> None:
        """Publish a message to an MQTT topic."""
        if not self._client:
            raise RuntimeError("MQTT client not initialized. Call connect() first.")
        self._client.publish(
            topic=topic,
            payload=json.dumps(payload, default=str).encode("utf-8"),
            qos=qos,
            retain=retain,
        )
        logger.debug("Published to MQTT topic: %s", topic)

    def disconnect(self) -> None:
        """Gracefully disconnect from MQTT broker."""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False
            logger.info("MQTT client disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected


# Singleton instance
mqtt_manager = MQTTManager()
