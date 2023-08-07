import React from 'react';

function DynamicSelect() {
    const [uf, setUf] = React.useState('AC');
    const [listUf, setListUf] = React.useState([]);
    function loadUf() {
        let url = 'https://servicodados.ibge.gov.br/';
        url = url + 'api/v1/localidades/estados';
        fetch(url)
          .then(response => response.json())
          .then(data => {        
            data.sort((a,b) => a.nome.localeCompare(b.nome));
            setListUf([...data]);
           });
    }
    React.useEffect(() => {
      loadUf();
    },[]);
    return (
      
        <select value={uf} onChange={e => setUf(e.target.value)} multiple={true}>
        {listUf.map((a, b) => ( 
            <option value={a.id}>{a.sigla} - {a.nome}</option>
         ))}
        </select>
      
    )
  }

  export default DynamicSelect;