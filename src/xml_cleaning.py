import re


def clean_xml_table(xml_string: str) -> str:
    """
    Remove unnecessary attributes and elements from XML table to create
    a compact pseudo-XML that preserves structure but reduces tokens.
    
    Args:
        xml_string: Raw XML table string
        
    Returns:
        Cleaned pseudo-XML string
    """
    
    # Step 1: Remove namespace declarations
    xml_string = re.sub(r'xmlns[^=]*="[^"]*"\s*', '', xml_string)
    
    # Step 2: Remove namespace prefixes from tags
    xml_string = re.sub(r'</?ns\d+:', '<', xml_string)
    xml_string = re.sub(r'</ns\d+:', '</', xml_string)
    
    # Step 3: Remove alignment attributes
    alignment_attrs = [
        r'halign="[^"]*"',
        r'valign="[^"]*"', 
        r'align="[^"]*"',
        r'char="[^"]*"',
        r'charoff="[^"]*"'
    ]
    for attr in alignment_attrs:
        xml_string = re.sub(attr + r'\s*', '', xml_string)
    
    # Step 4: Remove visual styling attributes
    visual_attrs = [
        r'frame="[^"]*"',
        r'rowsep="[^"]*"',
        r'colsep="[^"]*"',
        r'pgwide="[^"]*"',
        r'orient="[^"]*"',
        r'tabstyle="[^"]*"'
    ]
    for attr in visual_attrs:
        xml_string = re.sub(attr + r'\s*', '', xml_string)
    
    # Step 5: Remove sizing/spacing attributes
    sizing_attrs = [
        r'colwidth="[^"]*"',
        r'rowheight="[^"]*"',
        r'cellpadding="[^"]*"',
        r'cellspacing="[^"]*"'
    ]
    for attr in sizing_attrs:
        xml_string = re.sub(attr + r'\s*', '', xml_string)
    
    # Step 6: Remove ID and reference attributes
    reference_attrs = [
        r'id="[^"]*"',
        r'linkend="[^"]*"',
        r'xref="[^"]*"',
        r'href="[^"]*"'
    ]
    for attr in reference_attrs:
        xml_string = re.sub(attr + r'\s*', '', xml_string)
    
    # Step 7: Remove metadata attributes
    metadata_attrs = [
        r'role="[^"]*"',
        r'remap="[^"]*"',
        r'status="[^"]*"',
        r'revision="[^"]*"',
        r'xml:lang="[^"]*"',
        r'xml:space="[^"]*"'
    ]
    for attr in metadata_attrs:
        xml_string = re.sub(attr + r'\s*', '', xml_string)
    
    # Step 8: Remove CALS-specific redundant attributes
    cals_attrs = [
        r'colnum="[^"]*"',
        r'cols="[^"]*"'  # Remove from tgroup too since it's often redundant
    ]
    for attr in cals_attrs:
        xml_string = re.sub(attr + r'\s*', '', xml_string)
    
    # Step 9: Convert CALS spanning to standard HTML-style (optional)
    # nameend/namest/morerows -> colspan/rowspan
    xml_string = _convert_cals_spanning(xml_string)
    
    # Step 10: Remove entire colspec sections (they're usually redundant)
    xml_string = re.sub(r'<colspec[^>]*/?>\s*', '', xml_string)
    
    # Step 11: Remove unnecessary elements
    unnecessary_elements = [
        r'<alt-text[^>]*>.*?</alt-text>\s*',
        r'<alt-text[^>]*/>\s*'
    ]
    for elem in unnecessary_elements:
        xml_string = re.sub(elem, '', xml_string, flags=re.DOTALL)
    
    # Step 12: Clean up whitespace
    xml_string = re.sub(r'\s+>', '>', xml_string)  # Remove space before closing >

    # more compact syntax
    xml_string = re.sub(r'<entry ', '<', xml_string)
    xml_string = re.sub(r'<entry>', '<>', xml_string)
    xml_string = re.sub(r'colname', 'col', xml_string)
    xml_string = re.sub(r'"col(\d+)"', r'"\1"', xml_string)

    return xml_string.strip()

def _convert_cals_spanning(xml_string: str) -> str:
    """Convert CALS spanning attributes to HTML-style colspan/rowspan."""
    
    # Convert morerows to rowspan
    def replace_morerows(match):
        morerows = int(match.group(1))
        rowspan = morerows + 1
        return f'rowspan="{rowspan}"'
    
    xml_string = re.sub(r'morerows="(\d+)"', replace_morerows, xml_string)
    
    # Convert nameend/namest to colspan (basic conversion)
    # This is more complex and would need column name mapping for full accuracy
    # For now, just remove them as they're often redundant with explicit colspan
    xml_string = re.sub(r'nameend="[^"]*"\s*', '', xml_string)
    xml_string = re.sub(r'namest="[^"]*"\s*', '', xml_string)
    
    return xml_string


XML_CLEANUP_DESCRIPTION = """
The XML tables have been processed for a more compact table representation, so less tokens are used.

For example, a table that would've originally looked like this:

```original_xml
<ns0:table xmlns:ns0="http://url/xml/common/cals/dtd" xmlns:ns1="http://url/xml/common/cals/dtd" id="tbl0005" rowsep="0" colsep="0" frame="topbot">
<ns0:label>Table 5</ns0:label>
<ns0:caption id="cap0007">
<ns0:simple-para id="spara014" view="all">Output responses</ns0:simple-para>
</ns0:caption>
<ns0:alt-text id="alt0007" role="short">Table 5</ns0:alt-text>
<ns1:tgroup cols="11">
<ns1:colspec colnum="1" colname="col1" align="left" />
<ns1:colspec colnum="2" colname="col2" align="left" />
<ns1:colspec colnum="3" colname="col3" align="left" />
<ns1:thead>
<ns1:row rowsep="1">
<ns0:entry id="en0076" colname="col1" valign="top">S. No.</ns0:entry>
<ns0:entry id="en0077" colname="col2" valign="top">S</ns0:entry>
<ns0:entry id="en0078" colname="col3" valign="top">F</ns0:entry>
</ns1:row>
</ns1:thead>
<ns1:tbody>
<ns1:row>
<ns0:entry id="en0079" colname="col1" valign="top">1</ns0:entry>
<ns0:entry id="en0080" colname="col2" valign="top">1050</ns0:entry>
<ns0:entry id="en0081" colname="col3" valign="top">180</ns0:entry>
</ns1:row>
<ns1:row>
...
```

looks as follows after processing:

```processed_xml
<table>
<label>Table 5<label>
<caption>
<simple-para view="all">Output responses<simple-para>
<caption>
<alt-text>Table 5<alt-text>
<tgroup>
<thead>
<row>
<col="1">S. No.<>
<col="2">S<>
<col="3">F<>
<row>
<thead>
<tbody>
<row>
<col="1">1<>
<col="2">1050<>
<col="3">180<>
<row>
...
```

So do not get confused by the slighlty different pseudo-xml code that you'll see for the tables.
"""
