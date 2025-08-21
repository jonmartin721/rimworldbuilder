"""
Unit tests for RimWorld save file parser.
"""

import pytest
import xml.etree.ElementTree as ET
from unittest.mock import patch, mock_open

from src.parser.save_parser import RimWorldSaveParser as SaveParser
from src.models.game_entities import BuildingType, Position


class TestSaveParser:
    """Tests for SaveParser class"""

    @pytest.fixture
    def parser(self):
        """Create a parser instance"""
        return SaveParser()

    def test_parse_position_standard(self, parser):
        """Test parsing standard position format"""
        pos_str = "(100, 0, 150)"
        pos = parser._parse_cell_position(pos_str)
        assert pos.x == 100
        assert pos.y == 150  # z becomes y
        assert pos.z == 0

    def test_parse_position_invalid(self, parser):
        """Test parsing invalid position returns None"""
        assert parser._parse_cell_position("invalid") is None
        assert parser._parse_cell_position("(100, 150)") is None
        assert parser._parse_cell_position("") is None

    def test_is_building(self, parser):
        """Test building type detection"""
        assert parser._is_building("Wall_WoodPlank")
        assert parser._is_building("Door_Steel")
        assert parser._is_building("Bed_Wood")
        assert not parser._is_building("Tree_Oak")
        assert not parser._is_building("Human")

    def test_detect_building_type(self, parser):
        """Test building type classification"""
        assert parser._detect_building_type("Wall_WoodPlank") == BuildingType.WALL
        assert parser._detect_building_type("Door_Steel") == BuildingType.DOOR
        assert parser._detect_building_type("Bed_Wood") == BuildingType.BEDROOM
        assert parser._detect_building_type("ElectricStove") == BuildingType.KITCHEN
        assert parser._detect_building_type("DiningChair") == BuildingType.RECREATION
        assert parser._detect_building_type("Frame_HeavyBridge") == BuildingType.BRIDGE
        assert parser._detect_building_type("PowerConduit") == BuildingType.CONDUIT
        assert parser._detect_building_type("Unknown") == BuildingType.OTHER

    def test_parse_building_xml(self, parser):
        """Test parsing a building from XML"""
        xml_str = """
        <thing Class="Building">
            <def>Wall_WoodPlank</def>
            <id>wall_123</id>
            <pos>(10, 0, 20)</pos>
            <health>
                <maxHealth>300</maxHealth>
                <curHealth>250</curHealth>
            </health>
        </thing>
        """
        thing = ET.fromstring(xml_str)
        building = parser._parse_building(
            thing, "wall_123", "Wall_WoodPlank", Position(10, 20, 0)
        )

        assert building.id == "wall_123"
        assert building.def_name == "Wall_WoodPlank"
        assert building.position.x == 10
        assert building.position.y == 20
        assert building.building_type == BuildingType.WALL
        assert building.health == 250
        assert building.max_health == 300

    def test_parse_frame_building(self, parser):
        """Test parsing Frame class buildings (like bridges)"""
        xml_str = """
        <thing Class="Frame">
            <def>Frame_HeavyBridge</def>
            <id>frame_456</id>
            <pos>(50, 0, 60)</pos>
        </thing>
        """
        thing = ET.fromstring(xml_str)
        building = parser._parse_building(
            thing, "frame_456", "Frame_HeavyBridge", Position(50, 60, 0)
        )

        assert building.def_name == "Frame_HeavyBridge"
        assert building.building_type == BuildingType.BRIDGE
        assert building.is_frame

    def test_parse_colonist(self, parser):
        """Test parsing a colonist from XML"""
        xml_str = """
        <thing Class="Pawn">
            <def>Human</def>
            <id>Human123</id>
            <kindDef>Colonist</kindDef>
            <name Class="NameTriple">
                <first>John</first>
                <nick>Johnny</nick>
                <last>Doe</last>
            </name>
            <faction>Faction_PlayerColony</faction>
            <pos>(100, 0, 100)</pos>
            <health>100</health>
        </thing>
        """
        thing = ET.fromstring(xml_str)
        colonist = parser._parse_colonist(thing)

        assert colonist is not None
        assert colonist.id == "Human123"
        assert colonist.name == "John 'Johnny' Doe"
        assert colonist.position.x == 100
        assert colonist.position.y == 100
        assert colonist.faction == "Faction_PlayerColony"

    def test_parse_colonist_non_player_faction(self, parser):
        """Test that non-player faction pawns are not parsed as colonists"""
        xml_str = """
        <thing Class="Pawn">
            <def>Human</def>
            <faction>Faction_Empire</faction>
        </thing>
        """
        thing = ET.fromstring(xml_str)
        colonist = parser._parse_colonist(thing)
        assert colonist is None

    def test_extract_map_size(self, parser):
        """Test extracting map size from XML"""
        xml_str = """
        <map>
            <mapInfo>
                <size>(250, 1, 250)</size>
            </mapInfo>
        </map>
        """
        root = ET.fromstring(xml_str)
        parser.root = root
        size = parser._extract_map_size()
        assert size == (250, 250)

    def test_building_type_mapping(self, parser):
        """Test the building type mapping dictionary"""
        # Test exact matches
        assert "Wall" in parser.building_type_mapping[BuildingType.WALL]
        assert "Door" in parser.building_type_mapping[BuildingType.DOOR]
        assert "Bed" in parser.building_type_mapping[BuildingType.BEDROOM]

        # Test partial matches
        assert "Table" in parser.building_type_mapping[BuildingType.KITCHEN]
        assert "Chair" in parser.building_type_mapping[BuildingType.RECREATION]
        assert "Bridge" in parser.building_type_mapping[BuildingType.BRIDGE]
        assert "PowerConduit" in parser.building_type_mapping[BuildingType.CONDUIT]

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='<?xml version="1.0"?><savegame></savegame>',
    )
    def test_parse_empty_save(self, mock_file, parser):
        """Test parsing an empty save file"""
        result = parser.parse("test.rws")
        assert result.buildings == []
        assert result.colonists == []
        assert result.zones == []
        assert result.mods == []


class TestSaveParserIntegration:
    """Integration tests using sample save data"""

    def test_parse_sample_save_structure(self):
        """Test parsing a minimal save file structure"""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
        <savegame>
            <map>
                <mapInfo>
                    <size>(100, 1, 100)</size>
                </mapInfo>
                <things>
                    <thing Class="Building">
                        <def>Wall_WoodPlank</def>
                        <id>wall_1</id>
                        <pos>(10, 0, 10)</pos>
                    </thing>
                    <thing Class="Frame">
                        <def>Frame_HeavyBridge</def>
                        <id>bridge_1</id>
                        <pos>(20, 0, 20)</pos>
                    </thing>
                    <thing Class="Pawn">
                        <def>Human</def>
                        <id>Human1</id>
                        <kindDef>Colonist</kindDef>
                        <name Class="NameTriple">
                            <first>Test</first>
                            <nick>Tester</nick>
                            <last>Person</last>
                        </name>
                        <faction>Faction_PlayerColony</faction>
                        <pos>(30, 0, 30)</pos>
                    </thing>
                </things>
            </map>
            <mods>
                <li>
                    <packageId>Ludeon.RimWorld</packageId>
                    <name>Core</name>
                </li>
            </mods>
        </savegame>"""

        with patch("builtins.open", mock_open(read_data=xml_content)):
            parser = SaveParser()
            result = parser.parse("test.rws")

            # Check buildings
            assert len(result.buildings) == 2
            assert result.buildings[0].def_name == "Wall_WoodPlank"
            assert result.buildings[1].def_name == "Frame_HeavyBridge"
            assert result.buildings[1].building_type == BuildingType.BRIDGE

            # Check colonists
            assert len(result.colonists) == 1
            assert result.colonists[0].name == "Test 'Tester' Person"

            # Check mods
            assert len(result.mods) == 1
            assert result.mods[0].package_id == "Ludeon.RimWorld"
            assert result.mods[0].name == "Core"

            # Check map size
            assert result.map_size == (100, 100)
