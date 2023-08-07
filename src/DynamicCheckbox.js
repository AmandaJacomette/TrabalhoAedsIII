import React, { useEffect, useState } from 'react';

const resposta_da_requisicao = [{
        id: 1,
        checked: false,
        name: "Checkbox 1",
      },
      {
        id: 2,
        checked: false,
        name: "Checkbox 2",
      },
      {
        id: 3,
        checked: false,
        name: "Checkbox 3",
      },];

export default function DynamicCheckbox() {
    const [dinamico, setDinamico] = useState([]);

    useEffect(() => {
        setDinamico(resposta_da_requisicao);
      },[]);


    function handleCheckboxes(id) {
      setDinamico(dinamico.map(dinamico => dinamico.id === id ? {...dinamico, checked: !dinamico.checked} : dinamico))
    }

  return (
    <label>
      {dinamico?.map(({ id, checked, name }) => (
      <label >
        <label>
          <input
            type='checkbox'
            onPress={() => handleCheckboxes(id)}
            color='#009688'
          />
        </label>
        <label onPress={item => setDinamico(!dinamico)}> {name}</label>
      </label>
      ))}

    </label>
  );
}